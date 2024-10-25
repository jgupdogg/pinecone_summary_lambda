def clean_empty_arrays_and_objects(obj):
    keys_to_delete = []
    for key, value in obj.items():
        if isinstance(value, dict):
            clean_empty_arrays_and_objects(value)
            if not value:
                keys_to_delete.append(key)
        elif isinstance(value, list) and not value:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del obj[key]
