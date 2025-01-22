import asyncio
import json
import os

from aiohttp import ClientSession

from tool import tool

_URL = "https://api.openai.com/v1/chat/completions"
_HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv("OPENAI_API_KEY")}"}

@tool
def greet(name: str, repetitions: int | None = None) -> None:
    """
    Print a greeting to `name` a specified number of times. If `repetitions`
    is not provided, a single greeting is printed.

    :param name: The person or entity to greet.
    :param repetitions: Number of times to greet (defaults to 1 if None).
    """
    times = repetitions if repetitions is not None else 1
    for _ in range(times):
        print(f"Hello, {name}!")


class Greeter:
    @tool
    def greet(name: str, repetitions: int | None = None) -> None:
        """
        Print a greeting to `name` a specified number of times. If `repetitions`
        is not provided, a single greeting is printed.

        :param name: The person or entity to greet.
        :param repetitions: Number of times to greet (defaults to 1 if None).
        """
        times = repetitions if repetitions is not None else 1
        for _ in range(times):
            print(f"Hello, {name}!")


async def main():
    payload = {"messages": [{"role": "user", "content": "use the `greet` tool to greet someone a few times"}],
               "model": "gpt-4o-mini",
               "tools": [greet._tool]}

    print(f"attempting to use tool schema on function {json.dumps(greet._tool, indent=4)}...")

    async with ClientSession() as session:
        async with session.post(_URL, headers=_HEADERS, json=payload) as resp:
            if resp.status != 200:
                print(f"bad status code {resp.status}: text={await resp.text()}")
                exit(1)

            print(await resp.json())

    g = Greeter()
    payload = {"messages": [{"role": "user", "content": "use the `greet` tool to greet someone a few times"}],
               "model": "gpt-4o-mini",
               "tools": [g.greet._tool]}

    print(f"attempting to use tool schema on method {json.dumps(greet._tool, indent=4)}...")

    async with ClientSession() as session:
        async with session.post(_URL, headers=_HEADERS, json=payload) as resp:
            if resp.status != 200:
                print(f"bad status code {resp.status}: text={await resp.text()}")
                exit(1)

            print(await resp.json())


if __name__ == "__main__":
    asyncio.run(main())
