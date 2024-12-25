import json


class CustomJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, obj):
        # Custom encode for the top-level object
        return super().encode(self._process_obj(obj))

    def _process_obj(self, obj):
        if isinstance(obj, str):
            if obj.startswith("data:image/jpeg;base64,"):
                return "data:image/jpeg;base64,..."
            else:
                return self._process_json_string(obj)
        elif isinstance(obj, list):
            return [self._process_obj(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._process_obj(value) for key, value in obj.items()}
        return obj

    def _process_json_string(self, obj):
        try:
            parsed_obj = json.loads(obj)
            return self._process_obj(parsed_obj)
        except (json.JSONDecodeError, TypeError):
            return obj


def dumps(obj, *args, **kwargs):
    return json.dumps(obj, cls=CustomJSONEncoder, *args, **kwargs)
