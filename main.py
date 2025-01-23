from typing import Optional, List, Dict, Any, Type, Literal
from functools import wraps
import asyncio
import json
import os

from pydantic import BaseModel, Field
from docstring_parser import parse as parse_docstring
from aiohttp import ClientSession


###############################################################################
# 1) Tools to convert a Pydantic model into a "strict" JSON Schema for OpenAI
###############################################################################

def prune_disallowed_keys(schema: Any) -> None:
    """
    Remove fields that OpenAI strict mode doesn't allow:
      - default, title, examples, format, etc.
    Operates in-place, recursively.
    """
    if isinstance(schema, dict):
        for key in ["default", "title", "examples", "format", "description"]:  
            # We can remove 'description' from sub-properties if you like, 
            # though the docs say "description" is allowed. 
            # It's optional: if you want descriptions, keep it.
            if key in schema:
                schema.pop(key)
        # Recurse into dict values
        for v in schema.values():
            prune_disallowed_keys(v)
    elif isinstance(schema, list):
        for item in schema:
            prune_disallowed_keys(item)


def force_strict_mode(schema: Any) -> None:
    """
    Recursively:
      1) If 'type' == 'object', set 'additionalProperties' = false
      2) Ensure every property name is in 'required'
    """
    if isinstance(schema, dict):
        # If it's an object
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            props = schema.get("properties", {})
            required = schema.setdefault("required", [])
            # Add every property to 'required'
            for prop_name in props.keys():
                if prop_name not in required:
                    required.append(prop_name)

        # Recurse deeper
        for v in schema.values():
            force_strict_mode(v)

    elif isinstance(schema, list):
        for item in schema:
            force_strict_mode(item)


def build_strict_openai_schema(
    *,
    func_name: str,
    func_description: str,
    param_model: Type[BaseModel],
) -> dict:
    """
    1. Generate a pydantic JSON schema inlined (no refs).
    2. Prune disallowed fields.
    3. Enforce strict mode at every object node.
    4. Return OpenAI function-calling schema with 'strict': true.
    """
    # (A) Get a fully inlined schema from Pydantic v2 by specifying mode="serialization"
    raw_schema = param_model.model_json_schema(mode="serialization")

    # We'll assume raw_schema is like:
    # { type: "object", properties: {...}, ... }

    # (B) Clean up to match OpenAI's strict schema constraints
    prune_disallowed_keys(raw_schema)
    force_strict_mode(raw_schema)

    # (C) Build final "type": "function" object
    return {
        "type": "function",
        "function": {
            "name": func_name,
            "description": func_description,
            "strict": True,
            "parameters": raw_schema,  
            # raw_schema now includes "type":"object","properties":...,"required":..., "additionalProperties":false, etc.
        }
    }


###############################################################################
# 2) Decorator: attach an OpenAI function schema to `func.tool`
###############################################################################

def tool(param_model: Type[BaseModel]):
    """
    Decorator that:
      1. Expects a pydantic model describing parameters
      2. Uses docstring-parser to get short/long function descriptions
      3. Builds a 'strict' function-calling JSON schema
      4. Stores it in `func.tool`
    """
    def decorator(func):
        doc = func.__doc__ or ""
        parsed = parse_docstring(doc)
        short_desc = (parsed.short_description or "").strip()
        long_desc = (parsed.long_description or "").strip()
        if short_desc and long_desc:
            func_description = short_desc + "\n\n" + long_desc
        else:
            func_description = short_desc or long_desc
        if not func_description:
            func_description = f"{func.__name__} (no description)"

        schema = build_strict_openai_schema(
            func_name=func.__name__,
            func_description=func_description,
            param_model=param_model,
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.tool = schema
        return wrapper

    return decorator


###############################################################################
# 3) Example: define nested models
###############################################################################

class GetWeatherParams(BaseModel):
    location: str = Field(..., description="Location to get weather for")
    unit: Optional[str] = Field(None, description="Temp unit: 'C' or 'F' (null if unspecified)")

@tool(GetWeatherParams)
async def get_weather(location: str, unit: Optional[str]) -> Dict[str, Any]:
    """
    Fetch the weather for a given location.

    Extended docstring...
    """
    return {"location": location, "unit": unit, "temp": "65F"}


class Message(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Message author")
    content: str = Field(..., description="Message content")

class GetResponseParams(BaseModel):
    messages: List[Message] = Field(..., description="Conversation so far")
    # Could also do a nested model if we want 'tools' to be structured, 
    # but let's keep it simple. 
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Optional tool schemas")

@tool(GetResponseParams)
async def get_response(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Return a response from the LLM for the conversation in `messages`.
    Possibly supply a set of function tools it may call.
    """
    _URL = "https://api.openai.com/v1/chat/completions"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No OPENAI_API_KEY found.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "tools": tools,
    }

    async with ClientSession() as session:
        async with session.post(_URL, headers=headers, json=payload) as resp:
            if resp.status != 200:
                print(f"HTTP {resp.status} - {await resp.text()}")
                return
            output = await resp.json()
            print(json.dumps(output, indent=2))


    return output


###############################################################################
# 4) Minimal test calling the OpenAI API
###############################################################################

async def main():
    await get_response([{"role": "user", "content": "call get weather for New York City and use get_response to calculate 2+2"}], 
                       [get_weather.tool, get_response.tool])


if __name__ == "__main__":
    # show the final schemas
    print("=== get_weather.tool ===")
    print(json.dumps(get_weather.tool, indent=2))
    print("\n=== get_response.tool ===")
    print(json.dumps(get_response.tool, indent=2))

    asyncio.run(main())
