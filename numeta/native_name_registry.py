class NativeNameRegistry:
    def __init__(self):
        self.reserved_names: set[str] = set()

    def is_reserved(self, name: str) -> bool:
        return name in self.reserved_names

    def reserve(self, name: str) -> None:
        self.reserved_names.add(name)

    def reserve_many(self, names) -> None:
        self.reserved_names.update(names)

    def release(self, name: str) -> None:
        self.reserved_names.discard(name)

    def release_many(self, names) -> None:
        self.reserved_names.difference_update(names)


native_name_registry = NativeNameRegistry()
