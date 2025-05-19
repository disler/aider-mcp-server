# Dependency Injection Container

The Dependency Injection Container provides a centralized way to manage component dependencies and lifecycles in the Aider MCP Server. It helps reduce coupling between components and improves testability by allowing dependencies to be injected rather than hardcoded.

## Features

- **Singleton**, **Transient**, and **Scoped** lifetimes
- **Constructor injection** with automatic dependency resolution
- **Async support** for async initialization and cleanup
- **Interface-based** dependency registration
- **Circular dependency detection**
- **Type safety** with proper type hints
- **Factory functions** for complex initialization

## Basic Usage

### Creating a Container

```python
from aider_mcp_server.dependency_container import DependencyContainer

# Create a container (usually done at application startup)
async with DependencyContainer() as container:
    # Register and resolve services here
    pass  # Container will be closed automatically when exiting the context
```

### Registering Services

```python
# Register a singleton (one instance for the entire application)
await container.register_singleton(ILogger, implementation_type=Logger)

# Register a transient service (new instance each time it's resolved)
await container.register_transient(IDatabase, implementation_type=Database)

# Register a scoped service (one instance per scope)
await container.register_scoped(IUserRepository, implementation_type=UserRepository)

# Register with an existing instance
logger = Logger()
await container.register_singleton(ILogger, instance=logger)

# Register with a factory function
async def create_database(logger: ILogger) -> Database:
    database = Database(logger)
    await database.connect()
    return database

await container.register_singleton(IDatabase, factory=create_database)
```

### Resolving Services

```python
# Resolve a service
logger = await container.resolve(ILogger)
database = await container.resolve(IDatabase)

# Dependencies are automatically injected
# If UserService depends on ILogger and IDatabase, they will be automatically injected
user_service = await container.resolve(IUserService)
```

### Creating Scopes

```python
# Create a scoped container (useful for request scoping)
async with await container.create_scope() as scoped_container:
    # Resolve scoped services
    repository = await scoped_container.resolve(IUserRepository)
    
    # Each scope gets its own instance of scoped services
    # Singleton services are shared across all scopes
```

## Best Practices

1. **Register services at startup**: Initialize the container and register all services during application startup.

2. **Use interfaces**: Register services using interface types rather than concrete implementation types to promote loose coupling.

3. **Constructor injection**: Design your classes to receive dependencies through their constructors. This makes dependencies explicit and easier to test.

4. **Avoid service locator pattern**: Instead of injecting the container itself, inject specific dependencies.

5. **Prefer async context managers**: Use the `async with` statement to ensure proper cleanup of resources.

6. **Proper scope usage**:
   - Use **Singleton** for services that maintain state and should be shared across the application.
   - Use **Transient** for stateless services or services that should not share state.
   - Use **Scoped** for services that should maintain state within a specific scope (e.g., a request).

## Example

```python
from aider_mcp_server.dependency_container import DependencyContainer
from typing import Protocol

# Define interfaces
class ILogger(Protocol):
    def log(self, message: str) -> None: ...

class IDatabase(Protocol):
    async def query(self, sql: str) -> list: ...

# Implement services
class Logger:
    def log(self, message: str) -> None:
        print(f"LOG: {message}")

class Database:
    def __init__(self, logger: ILogger):
        self.logger = logger
        
    async def query(self, sql: str) -> list:
        self.logger.log(f"Executing query: {sql}")
        return [{"result": "ok"}]

class UserRepository:
    def __init__(self, database: IDatabase, logger: ILogger):
        self.database = database
        self.logger = logger
        
    async def get_user(self, user_id: str) -> dict:
        self.logger.log(f"Getting user {user_id}")
        return (await self.database.query(f"SELECT * FROM users WHERE id = '{user_id}'"))[0]

# Application startup
async def main():
    async with DependencyContainer() as container:
        # Register services
        await container.register_singleton(ILogger, implementation_type=Logger)
        await container.register_singleton(IDatabase, implementation_type=Database)
        await container.register_scoped(UserRepository, implementation_type=UserRepository)
        
        # Create a scope (e.g., for a request)
        async with await container.create_scope() as request_scope:
            # Resolve the repository (with all dependencies automatically injected)
            repo = await request_scope.resolve(UserRepository)
            
            # Use the repository
            user = await repo.get_user("user123")
            print(user)
```

## Integration with Existing Code

To integrate the DI container with the existing codebase:

1. Initialize the container in `multi_transport_server.py` or similar startup code.
2. Register core services during initialization.
3. Replace the existing `ComponentInitializer` with container-based initialization.
4. Inject dependencies into components through their constructors.
5. Use scoped containers for request handling.

This approach will gradually reduce coupling and improve testability throughout the codebase.