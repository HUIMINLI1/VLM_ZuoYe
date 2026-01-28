'''
$lhm 251024
'''
with open('cfg/server_config.yaml', 'r', encoding='utf-8') as f:
    import yaml
    config = yaml.safe_load(f)

with open('set_env.bat', 'w', encoding='utf-8') as f:
    for k,v in config.items():
        if isinstance(v, str | int | float):
            f.write(f'set {k}={v}\n')
        elif isinstance(v, bool):
            f.write(f'set {k}={str(v).lower()}\n')
        elif isinstance(v, list):
            f.write(f'set {k}={",".join(map(str, v))}\n')
