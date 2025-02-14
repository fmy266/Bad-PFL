
class Event:
    def __init__(self):
        self.handlers = []

    def register(self, handler):
        self.handlers.append(handler)

    def unregister(self, handler):
        self.handlers.remove(handler)

    def trigger(self, *args, **kwargs):
        for handler in self.handlers:
            handler(*args, **kwargs)


class EventEmitter:
    def __init__(self):
        self._events = {}
        self.state_info = {}

    def on(self, event_name, handler):
        if event_name not in self._events:
            self._events[event_name] = Event()
        self._events[event_name].register(handler)

    def off(self, event_name, handler):
        if event_name in self._events:
            self._events[event_name].unregister(handler)

    def emit(self, event_name, *args, **kwargs):
        if event_name in self._events:
            self._events[event_name].trigger(*args, **kwargs)

    def __call__(self, event_name):
        def decorator(handler):
            self.on(event_name, handler)
            return handler
        return decorator


fl_event_emitter = EventEmitter()






