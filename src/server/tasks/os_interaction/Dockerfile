FROM python:3.10

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install agentrl-worker

RUN python - <<'PY'
import pathlib

p = pathlib.Path("/usr/local/lib/python3.10/site-packages/agentrl/worker/environment/docker.py")
lines = p.read_text().splitlines(True)

out = []
inserted = False

for line in lines:
    # 找到 create 调用行，并拿到它的缩进
    if (not inserted) and ("client.containers.create" in line and "attrs" in line and "container_name" in line):
        indent = line[:len(line) - len(line.lstrip())]
        out.append(indent + 'host_cfg = attrs.get("HostConfig") or {}\n')
        out.append(indent + 'for k in ("NanoCPUs","NanoCpus","CpuPeriod","CpuQuota","CpuShares"):\n')
        out.append(indent + '    host_cfg.pop(k, None)\n')
        out.append(indent + 'attrs["HostConfig"] = host_cfg\n')
        inserted = True
    out.append(line)

if not inserted:
    raise SystemExit(f"patch point not found in {p}")

p.write_text("".join(out))
print("patched", p)
PY

# 可选：构建时就验证语法没问题
RUN python -m py_compile /usr/local/lib/python3.10/site-packages/agentrl/worker/environment/docker.py

RUN python - <<'PY'
import pathlib

p = pathlib.Path("/usr/local/lib/python3.10/site-packages/agentrl/worker/environment/docker.py")
txt = p.read_text().splitlines(True)

# 幂等：如果已经 patch 过就不再重复插入
if "ClientTimeout(total=timeout)" in "".join(txt):
    print("timeout patch already present")
    raise SystemExit(0)

out = []
inserted = False

for line in txt:
    # 找到 aiodocker exec start 那一行（你日志里对应 docker.py:259）
    if (not inserted) and ("exec_.start(" in line and "timeout=timeout" in line):
        indent = line[:len(line) - len(line.lstrip())]
        out.append(indent + "import aiohttp\n")
        out.append(indent + "if isinstance(timeout, (int, float)):\n")
        out.append(indent + "    timeout = aiohttp.ClientTimeout(total=timeout)\n")
        inserted = True
    out.append(line)

if not inserted:
    raise SystemExit(f"patch point not found in {p}")

p.write_text("".join(out))
print("patched timeout:", p)
PY

RUN python -m py_compile /usr/local/lib/python3.10/site-packages/agentrl/worker/environment/docker.py

COPY ./src/server/tasks/os_interaction /app/src/server/tasks/os_interaction
COPY ./configs/tasks/os.yaml /app/configs/tasks/os.yaml
COPY ./data/os_interaction /app/data/os_interaction

COPY --chmod=0755 ./extra/worker-entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh", "-c", "configs/tasks/os.yaml"]
