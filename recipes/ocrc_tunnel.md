# ocrc SSH tunnel recipe

The ocrc PDF-parsing service runs on the GPU server bound to localhost:8601
only. Expose it locally on :18601 before using ocrc:

```sh
# kill any stale tunnel, bring up a fresh one
pkill -f "ssh.*-L 18601:127.0.0.1:8601" 2>/dev/null
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -f -N \
    -L 18601:127.0.0.1:8601 user@192.168.1.68
curl -s --max-time 3 http://127.0.0.1:18601/api/v1/queue  # should print JSON
```

Then parse a paper (stdout = zip archive; unzip it):

```sh
mkdir -p papers/<ARXIV_ID> && cd papers/<ARXIV_ID>
OCRC_SERVER=http://127.0.0.1:18601 ocrc parse https://arxiv.org/pdf/<ARXIV_ID> \
    --quiet 1>/tmp/p.zip 2>parse.log
unzip -o /tmp/p.zip          # -> document.md, images/, layout/, meta.json
```

Parse is slow (~10-30s/page); queue + cache on the server side, so repeat calls
are fast.
