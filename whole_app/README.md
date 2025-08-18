# Whole App

### Understanding Async Methods

We uses Python's `asyncio` library to handle operations that might take time (like network requests, disk I/O, or calling native methods) without blocking the entire user interface.

- `async def`: Defines a coroutine. This is a function that can be paused and resumed. When a coroutine needs to wait for something (like a network response or a method call to Dart), it `await`s.
- `await`: Used inside an `async def` function. It pauses the execution of the current coroutine until the awaited operation completes. While it's paused, the Flet event loop can switch to running other parts of app's UI or other tasks, keeping the UI responsive.

