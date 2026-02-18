"""HTTP server for handling webhook requests from various channels."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import aiohttp.web
from loguru import logger

if TYPE_CHECKING:
    from nanobot.channels.manager import ChannelManager


class GatewayServer:
    """
    HTTP server that handles incoming webhook requests from chat platforms.
    
    Routes:
        POST /webhook/ha - Home Assistant webhook endpoint
        GET /health - Health check endpoint
    """
    
    def __init__(self, channel_manager: ChannelManager, host: str = "0.0.0.0", port: int = 18790):
        """
        Initialize the gateway server.
        
        Args:
            channel_manager: Channel manager instance for routing requests
            host: Host to bind to (default: 0.0.0.0 for all interfaces)
            port: Port to listen on (default: 18790)
        """
        self.channel_manager = channel_manager
        self.host = host
        self.port = port
        self.app = aiohttp.web.Application()
        self.runner: aiohttp.web.AppRunner | None = None
        self.site: aiohttp.web.TCPSite | None = None
        
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Configure HTTP routes."""
        self.app.router.add_post("/webhook/ha", self._handle_ha_webhook)
        self.app.router.add_get("/health", self._handle_health)
        
        # Log all registered routes
        logger.debug("Registered routes:")
        for route in self.app.router.routes():
            logger.debug(f"  {route.method} {route.resource}")
    
    async def _handle_ha_webhook(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """
        Handle incoming webhook from Home Assistant.
        
        Expected JSON payload:
        {
            "sender_id": "user_name_or_entity",
            "conversation_id": "unique-id",
            "text": "turn on the kitchen lights",
            "metadata": {}  # optional
        }
        """
        try:
            data = await request.json()
            logger.info(
                "Received HA webhook: sender={}, conversation={}, text={!r}",
                data.get("sender_id"),
                data.get("conversation_id"),
                data.get("text", "")[:50],
            )
            
            # Get the Home Assistant channel
            ha_channel = self.channel_manager.get_channel("home_assistant")
            if not ha_channel:
                logger.error("Home Assistant channel not found or not enabled")
                return aiohttp.web.json_response(
                    {"error": "Home Assistant channel not enabled"},
                    status=503,
                )
            
            # Forward to the channel's webhook handler
            await ha_channel.handle_inbound_webhook(data)
            
            return aiohttp.web.json_response({"status": "ok"})
            
        except aiohttp.web.HTTPException:
            raise
        except Exception as exc:
            logger.error("Error handling HA webhook: {exc}", exc=exc)
            return aiohttp.web.json_response(
                {"error": str(exc)},
                status=500,
            )
    
    async def _handle_health(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Health check endpoint."""
        return aiohttp.web.json_response({
            "status": "ok",
            "channels": self.channel_manager.enabled_channels,
        })
    
    async def start(self) -> None:
        """Start the HTTP server."""
        logger.info(f"Starting gateway server on {self.host}:{self.port}")
        
        self.runner = aiohttp.web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = aiohttp.web.TCPSite(
            self.runner,
            self.host,
            self.port,
        )
        await self.site.start()
        
        logger.success(f"Gateway server listening on http://{self.host}:{self.port}")
        logger.info("Webhook endpoints:")
        logger.info(f"  - POST http://{self.host}:{self.port}/webhook/ha (Home Assistant)")
        logger.info(f"  - GET  http://{self.host}:{self.port}/health (Health check)")
    
    async def stop(self) -> None:
        """Stop the HTTP server."""
        logger.info("Stopping gateway server...")
        
        if self.site:
            await self.site.stop()
        
        if self.runner:
            await self.runner.cleanup()
        
        logger.info("Gateway server stopped")
    
    async def run_forever(self) -> None:
        """Run the server until interrupted."""
        await self.start()
        
        # Keep running until cancelled
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await self.stop()
            raise
