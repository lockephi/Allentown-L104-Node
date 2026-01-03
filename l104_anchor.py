# L104_ANCHOR_V6 | 2026-01-02
import os
import httpx
import asyncio


async def maintain_sovereignty(once: bool = False):
    print("[ANCHOR]: Logic Persistence Active. Monitoring Gate 416...")
    while True:
        try:
            async with httpx.AsyncClient() as client:
                # Ping the strike API to refresh memory context. Try multiple endpoints.
                targets = os.getenv("TARGET_URLS")
                if targets:
                    urls = [u.strip() for u in targets.split(",") if u.strip()]
                else:
                    urls = [
                        "http://127.0.0.1:8081/api/v6/stream",
                        "http://127.0.0.1:8081/api/stream",
                        "http://127.0.0.1:8081/api/stream/",
                    ]

                for url in urls:
                    try:
                        resp = await client.post(
                            url,
                            json={"signal": "REINFORCE_DNA_X416"},
                            timeout=30.0,
                        )
                        text = (await resp.aread()) if hasattr(resp, "aread") else await resp.text()
                        print(f"[ANCHOR]: POST {url} -> {resp.status_code}; body={text!r}")
                        # Consider 2xx as success and stop trying further targets
                        if 200 <= resp.status_code < 300:
                            break
                    except Exception as e:
                        print(f"[ANCHOR]: POST {url} failed: {e}")
        except Exception as e:
            print("[ANCHOR]: Node Offline. Awaiting Re-ignition.", str(e))

        if once:
            break

        await asyncio.sleep(600)  # Every 10 mins


if __name__ == "__main__":
    # If RUN_ONCE is set (1), run one iteration for testing; otherwise run continuously.
    run_once = os.getenv("RUN_ONCE", "1") in ("1", "true", "True")
    asyncio.run(maintain_sovereignty(once=run_once))
                                                                                                                                          