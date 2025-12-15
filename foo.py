import torch

payload = torch.load(
    "/home/rzuo02/work/sb3/rl-baselines3-zoo/logs/breakout_values.pt",
    map_location="cpu",
    weights_only=False,
)

print(len(payload["records"]), payload["metadata"])
print(f"{payload['records'][0].keys()=}")
print(f"{payload['records'][0]['observation'].shape=}")
print(f"{payload['records'][0]['value_prediction']=}")
print(f"{payload['records'][0]['returns']=}")

