"""Home Assistant conversation channel for nanobot."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus

from .base import BaseChannel


@dataclass
class HAChannelConfig:
    """
    Configuration for the Home Assistant channel.

    Attributes:
        ha_url: Base URL of the Home Assistant instance (e.g. http://homeassistant.local:8123).
        ha_token: Long-lived access token for authentication.
        webhook_id: Webhook ID registered in the custom component.
            Used by HA to push responses back to nanobot.
        nanobot_webhook_url: URL where nanobot listens for HA responses
            (e.g. http://nanobot-host:8080/ha/response).
        agent_id: Optional HA conversation agent ID. Defaults to the built-in agent.
        language: BCP-47 language tag forwarded to HA (e.g. "en").
        poll_interval: Seconds between health-check polls. 0 disables polling.
        allow_from: Optional list of sender IDs permitted to use this channel.
            Empty list means everyone is allowed.
        request_timeout: HTTP request timeout in seconds.
    """

    ha_url: str
    ha_token: str
    webhook_id: str
    nanobot_webhook_url: str
    agent_id: str = "conversation.home_assistant"
    language: str = "en"
    poll_interval: float = 30.0
    allow_from: list[str] = field(default_factory=list)
    request_timeout: int = 30


class HAChannel(BaseChannel):
    """
    Channel that bridges nanobot with Home Assistant's conversation API.

    Inbound flow (HA → nanobot):
        HA custom component POSTs a user utterance to nanobot's webhook
        → HAChannel receives it and calls _handle_message()
        → message enters the nanobot bus for the agent to process.

    Outbound flow (nanobot → HA):
        Agent publishes an OutboundMessage on the bus
        → HAChannel.send() forwards the text to the HA custom component
          via the /api/webhook/<webhook_id> endpoint
        → HA component delivers the response to the originating context.
    """

    name: str = "home_assistant"

    def __init__(self, config: HAChannelConfig, bus: MessageBus) -> None:
        super().__init__(config, bus)
        self.config: HAChannelConfig = config

        self._session: aiohttp.ClientSession | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._inbound_server: asyncio.Server | None = None

        # Maps conversation_id → asyncio.Event so send() can await HA's ack.
        self._pending: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the channel: open HTTP session and launch background tasks."""
        if self._running:
            logger.warning("HAChannel.start() called while already running.")
            return

        logger.info(
            "Starting HAChannel — HA URL: {url}, agent: {agent}",
            url=self.config.ha_url,
            agent=self.config.agent_id,
        )

        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        self._session = aiohttp.ClientSession(
            headers=self._auth_headers(),
            timeout=timeout,
        )

        self._running = True

        # Register this nanobot instance with the HA component so it knows
        # where to POST inbound messages.
        await self._register_with_ha()

        if self.config.poll_interval > 0:
            self._poll_task = asyncio.create_task(
                self._health_poll_loop(),
                name="ha_channel_health_poll",
            )

        logger.success("HAChannel started.")

    async def stop(self) -> None:
        """Gracefully stop the channel and release resources."""
        logger.info("Stopping HAChannel…")
        self._running = False

        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        if self._session and not self._session.closed:
            await self._session.close()

        logger.info("HAChannel stopped.")

    # ------------------------------------------------------------------
    # Outbound: nanobot → Home Assistant
    # ------------------------------------------------------------------

    async def send(self, msg: OutboundMessage) -> None:
        """
        Forward an agent response to the HA custom component.

        The component is responsible for delivering the text to the user
        (e.g. via TTS, a lovelace notification, or an automation trigger).

        Args:
            msg: Outbound message produced by the nanobot agent.
        """
        if not self._session:
            logger.error("HAChannel.send() called before start().")
            return

        payload = self._build_response_payload(msg)
        url = self._webhook_url()

        logger.debug(
            "Sending response to HA webhook {url}: {payload}",
            url=url,
            payload=payload,
        )

        try:
            async with self._session.post(url, json=payload) as resp:
                resp.raise_for_status()
                logger.info(
                    "Response delivered to HA (conversation_id={cid}, status={s})",
                    cid=msg.chat_id,
                    s=resp.status,
                )
        except aiohttp.ClientResponseError as exc:
            logger.error(
                "HA webhook returned HTTP {status}: {msg}",
                status=exc.status,
                msg=exc.message,
            )
        except aiohttp.ClientError as exc:
            logger.error("Network error sending to HA: {exc}", exc=exc)

    # ------------------------------------------------------------------
    # Inbound: Home Assistant → nanobot
    # (called by the nanobot web server / webhook router)
    # ------------------------------------------------------------------

    async def handle_inbound_webhook(self, data: dict[str, Any]) -> None:
        """
        Entry point for messages arriving from the HA custom component.

        The HA component POSTs JSON with at least:
            {
                "sender_id":      "user_name_or_entity",
                "conversation_id": "unique-id",
                "text":           "turn on the kitchen lights",
                "metadata":       {}          # optional
            }

        This method is called by your web framework's route handler.

        Args:
            data: Parsed JSON body from the HA component.
        """
        sender_id = data.get("sender_id", "ha_user")
        conversation_id = data.get("conversation_id", "default")
        text = data.get("text", "").strip()
        metadata = data.get("metadata", {})

        if not text:
            logger.warning("Received empty inbound message from HA; ignoring.")
            return

        logger.debug(
            "Inbound from HA — sender={sid}, conversation={cid}, text={t!r}",
            sid=sender_id,
            cid=conversation_id,
            t=text,
        )

        await self._handle_message(
            sender_id=sender_id,
            chat_id=conversation_id,
            content=text,
            metadata={
                "source": "home_assistant",
                "agent_id": self.config.agent_id,
                **metadata,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.ha_token}",
            "Content-Type": "application/json",
        }

    def _webhook_url(self) -> str:
        """URL on the HA side that receives nanobot responses."""
        base = self.config.ha_url.rstrip("/")
        return f"{base}/api/webhook/{self.config.webhook_id}"

    def _build_response_payload(self, msg: OutboundMessage) -> dict[str, Any]:
        """Serialize an OutboundMessage into the HA webhook payload."""
        return {
            "conversation_id": msg.chat_id,
            "sender_id": msg.recipient_id,
            "text": msg.content,
            "metadata": msg.metadata or {},
        }

    async def _register_with_ha(self) -> None:
        """
        Notify the HA component of this nanobot instance's callback URL.

        HA stores the URL and uses it to POST future inbound messages.
        Retries up to 3 times on failure so transient startup errors are
        handled gracefully.
        """
        if not self._session:
            return

        url = f"{self.config.ha_url.rstrip('/')}/api/nanobot_agent/register"
        payload = {
            "channel": self.name,
            "callback_url": self.config.nanobot_webhook_url,
            "agent_id": self.config.agent_id,
            "webhook_id": self.config.webhook_id,
        }

        # DIAGNOSTIC: Log the exact request details
        logger.info("=" * 60)
        logger.info("DIAGNOSTIC: HA Registration Request Details")
        logger.info("=" * 60)
        logger.info(f"Registration URL: {url}")
        logger.info(f"Request Payload: {payload}")
        logger.info(f"Session Headers: {dict(self._session.headers)}")
        logger.info("=" * 60)

        for attempt in range(1, 4):
            try:
                async with self._session.post(url, json=payload) as resp:
                    # DIAGNOSTIC: Log response details before raising
                    response_text = await resp.text()
                    logger.info("=" * 60)
                    logger.info(f"DIAGNOSTIC: HA Response (Attempt {attempt})")
                    logger.info("=" * 60)
                    logger.info(f"Status Code: {resp.status}")
                    logger.info(f"Response Headers: {dict(resp.headers)}")
                    logger.info(f"Response Body: {response_text}")
                    logger.info("=" * 60)
                    
                    resp.raise_for_status()
                    logger.info(
                        "Registered with HA successfully (attempt {n}).",
                        n=attempt,
                    )
                    return
            except aiohttp.ClientResponseError as exc:
                logger.error("=" * 60)
                logger.error(f"DIAGNOSTIC: Registration Failed (Attempt {attempt})")
                logger.error("=" * 60)
                logger.error(f"HTTP Status: {exc.status}")
                logger.error(f"Error Message: {exc.message}")
                logger.error(f"Request URL: {exc.request_info.url}")
                logger.error(f"Request Headers: {dict(exc.request_info.headers)}")
                logger.error("=" * 60)
                logger.warning(
                    "Registration attempt {n}/3 failed: {exc}",
                    n=attempt,
                    exc=exc,
                )
                if attempt < 3:
                    await asyncio.sleep(2**attempt)  # exponential back-off
            except aiohttp.ClientError as exc:
                logger.error("=" * 60)
                logger.error(f"DIAGNOSTIC: Network Error (Attempt {attempt})")
                logger.error("=" * 60)
                logger.error(f"Error Type: {type(exc).__name__}")
                logger.error(f"Error Details: {exc}")
                logger.error("=" * 60)
                logger.warning(
                    "Registration attempt {n}/3 failed: {exc}",
                    n=attempt,
                    exc=exc,
                )
                if attempt < 3:
                    await asyncio.sleep(2**attempt)  # exponential back-off

        logger.error(
            "Could not register with HA after 3 attempts. "
            "Inbound messages will not arrive until registration succeeds."
        )

    async def _health_poll_loop(self) -> None:
        """Periodically ping HA's /api/ endpoint to detect connectivity loss."""
        url = f"{self.config.ha_url.rstrip('/')}/api/"
        while self._running:
            await asyncio.sleep(self.config.poll_interval)
            if not self._session or self._session.closed:
                break
            try:
                async with self._session.get(url) as resp:
                    if resp.status == 200:
                        logger.debug("HA health check OK.")
                    else:
                        logger.warning(
                            "HA health check returned status {s}.",
                            s=resp.status,
                        )
            except aiohttp.ClientError as exc:
                logger.warning("HA health check failed: {exc}", exc=exc)