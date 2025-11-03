import logging
import mcp.server.stdio
from typing import Annotated
from mcp.server import Server
from .requestor import request_data
from pydantic import BaseModel, Field
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    TextContent,
    Tool,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)


logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

server = Server("minima")

class Query(BaseModel):
    text: Annotated[
        str,
        Field(description="context to find")
    ]

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="minima-query",
            description="Find context in local files with optional filtering by file type, language, date, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by file type: code, doc, data, spreadsheet, presentation"
                    },
                    "file_extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by extension: .py, .pdf, .md, etc."
                    },
                    "languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by language: python, javascript, java, etc."
                    },
                    "directories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by directory path"
                    },
                    "max_chunk_index": {
                        "type": "integer",
                        "description": "Only return first N chunks (0 = first chunk only)"
                    },
                    "modified_after": {
                        "type": "string",
                        "description": "ISO timestamp - files modified after this date"
                    }
                },
                "required": ["text"]
            }
        )
    ]
    
@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    logging.info("List of prompts")
    return [
        Prompt(
            name="minima-query",
            description="Find a context in a local files",
            arguments=[
                PromptArgument(
                    name="context", description="Context to search", required=True
                )
            ]
        )            
    ]
    
@server.call_tool()
async def call_tool(name, arguments: dict) -> list[TextContent]:
    if name != "minima-query":
        logging.error(f"Unknown tool: {name}")
        raise ValueError(f"Unknown tool: {name}")

    logging.info("Calling tools")

    # Extract query and filters
    context = arguments.get("text")
    filters = {
        "file_types": arguments.get("file_types"),
        "file_extensions": arguments.get("file_extensions"),
        "languages": arguments.get("languages"),
        "directories": arguments.get("directories"),
        "max_chunk_index": arguments.get("max_chunk_index"),
        "modified_after": arguments.get("modified_after"),
    }

    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}

    if not context:
        logging.error("Context is required")
        raise McpError(INVALID_PARAMS, "Context is required")

    output = await request_data(context, filters)
    if "error" in output:
        logging.error(output["error"])
        raise McpError(INTERNAL_ERROR, output["error"])

    logging.info(f"Get prompt: {output}")
    result_data = output['result']
    output_text = result_data['output']
    links = result_data.get('links', [])
    metadata = result_data.get('metadata', [])
    filters_applied = result_data.get('filters_applied', {})

    # Format response with sources and metadata
    response_parts = []

    # Show applied filters
    if any(filters_applied.values()):
        filters_text = ", ".join([f"{k}={v}" for k, v in filters_applied.items() if v])
        response_parts.append(f"**Filters Applied:** {filters_text}")

    # Show sources
    if links:
        links_formatted = "\n".join([f"- {link}" for link in links])
        response_parts.append(f"\n**Sources ({len(links)}):**\n{links_formatted}")

    # Show metadata (first 3 results)
    if metadata:
        metadata_formatted = "\n".join([
            f"- {m['file_name']} ({m['file_type']}, chunk {m['chunk']})"
            for m in metadata[:3]
        ])
        response_parts.append(f"\n**Top Results:**\n{metadata_formatted}")

    response_parts.append(f"\n**Answer:**\n{output_text}")
    full_response = "\n".join(response_parts)

    result = []
    result.append(TextContent(type="text", text=full_response))
    return result
    
@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
    if not arguments or "context" not in arguments:
        logging.error("Context is required")
        raise McpError(INVALID_PARAMS, "Context is required")
        
    context = arguments["text"]

    output = await request_data(context)
    if "error" in output:
        error = output["error"]
        logging.error(error)
        return GetPromptResult(
            description=f"Faild to find a {context}",
            messages=[
                PromptMessage(
                    role="user", 
                    content=TextContent(type="text", text=error),
                )
            ]
        )

    logging.info(f"Get prompt: {output}")    
    output = output['result']['output']
    return GetPromptResult(
        description=f"Found content for this {context}",
        messages=[
            PromptMessage(
                role="user", 
                content=TextContent(type="text", text=output)
            )
        ]
    )

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="minima",
                server_version="0.0.1",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
