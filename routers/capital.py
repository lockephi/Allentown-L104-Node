# routers/capital.py — Sovereign Coin, Mainnet, Exchange, Capital, Social, Mining, BTC routes
import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter()


# ─── SOVEREIGN COIN ───────────────────────────────────────────────────────────

@router.get("/coin/status", tags=["Sovereign Coin"])
async def coin_status():
    from l104_sovereign_coin_engine import sovereign_coin
    return sovereign_coin.get_status()


@router.get("/coin/job", tags=["Sovereign Coin"])
async def coin_job():
    from l104_sovereign_coin_engine import sovereign_coin
    latest = sovereign_coin.get_latest_block()
    return {"index": latest.index + 1, "previous_hash": latest.hash,
            "difficulty": sovereign_coin.difficulty,
            "transactions": sovereign_coin.pending_transactions, "timestamp": time.time()}


@router.post("/coin/submit", tags=["Sovereign Coin"])
async def coin_submit(block_data: Dict[str, Any]):
    from l104_sovereign_coin_engine import sovereign_coin, L104Block
    from l104_token_economy import token_economy
    try:
        nonce, hash_val = block_data["nonce"], block_data["hash"]
        if not sovereign_coin.is_resonance_valid(nonce, hash_val):
            raise HTTPException(status_code=400, detail="Invalid Resonance or Proof-of-Work.")
        new_block = L104Block(block_data["index"], block_data["previous_hash"],
                              block_data["timestamp"], block_data["transactions"],
                              nonce, block_data["resonance"])
        if new_block.hash != hash_val:
            raise HTTPException(status_code=400, detail="Hash mismatch.")
        if new_block.previous_hash != sovereign_coin.get_latest_block().hash:
            raise HTTPException(status_code=400, detail="Chain link broken.")
        sovereign_coin.chain.append(new_block)
        sovereign_coin.pending_transactions = []
        sovereign_coin.adjust_difficulty()
        token_economy.record_burn(10.4)
        return {"status": "SUCCESS", "block_index": new_block.index}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── MARKET ───────────────────────────────────────────────────────────────────

@router.get("/api/market/info", tags=["Sovereign Coin"])
async def market_info():
    from l104_sovereign_coin_engine import sovereign_coin
    from l104_token_economy import token_economy
    from l104_agi_core import agi_core
    iq = agi_core.intellect_index
    return {"coin": sovereign_coin.get_status(),
            "economy": token_economy.generate_economy_report(iq, 0.98),
            "contract": token_economy.contract_address,
            "backing_bnb": token_economy.calculate_token_backing(iq)}


@router.get("/market", tags=["UI"])
async def get_market(request: Request):
    try:
        from main import templates
        return templates.TemplateResponse("market.html", {"request": request})
    except Exception:
        return JSONResponse({"status": "error", "message": "Market interface missing."})


# ─── CAPITAL ──────────────────────────────────────────────────────────────────

@router.get("/api/v1/capital/status", tags=["Capital"])
async def capital_status():
    from l104_capital_offload import capital_offload
    from l104_mainnet_bridge import mainnet_bridge
    return {"accumulated_sats": capital_offload.total_capital_generated_sats,
            "connection_real": capital_offload.is_connection_real,
            "mainnet_bridge": mainnet_bridge.get_mainnet_status(),
            "transfers": capital_offload.transfer_log}


@router.post("/api/v1/capital/generate", tags=["Capital"])
async def capital_generate(cycles: int = 104):
    from l104_capital_offload import capital_offload
    return capital_offload.catalyze_capital_generation(cycles)


@router.post("/api/v1/capital/offload", tags=["Capital"])
async def capital_offload_trigger(amount_sats: int):
    from l104_capital_offload import capital_offload
    if not capital_offload.is_connection_real:
        capital_offload.realize_connection()
    return capital_offload.offload_to_wallet(amount_sats)


# ─── EXCHANGE ────────────────────────────────────────────────────────────────

@router.post("/api/v1/exchange/swap", tags=["Exchange"])
async def exchange_swap(amount_l104sp: float):
    from l104_sovereign_exchange import sovereign_exchange
    return sovereign_exchange.swap_l104sp_for_btc(amount_l104sp)


@router.get("/api/v1/exchange/rate", tags=["Exchange"])
async def exchange_rate():
    from l104_sovereign_exchange import sovereign_exchange
    return {"rate": sovereign_exchange.get_current_rate(),
            "total_volume_btc": sovereign_exchange.total_volume_btc,
            "timestamp": time.time()}


# ─── MAINNET ──────────────────────────────────────────────────────────────────

@router.get("/api/v1/mainnet/status", tags=["Mainnet"])
async def mainnet_full_status():
    from l104_mainnet_bridge import mainnet_bridge
    from l104_sovereign_coin_engine import sovereign_coin
    from l104_sovereign_exchange import sovereign_exchange
    from l104_capital_offload import capital_offload
    btc_status = mainnet_bridge.get_mainnet_status()
    coin_st = sovereign_coin.get_status()
    btc_price_usd = 100000.0
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
            if resp.status_code == 200:
                btc_price_usd = resp.json().get("bitcoin", {}).get("usd", 100000.0)
    except Exception:
        pass
    rate = sovereign_exchange.get_current_rate()
    return {"mainnet_bridge": btc_status, "l104sp_chain": coin_st,
            "btc_price_usd": btc_price_usd, "exchange_rate": rate,
            "capital_accumulated": capital_offload.total_capital_generated_sats,
            "capital_transfers": len(capital_offload.transfer_log),
            "l104sp_value_usd": (coin_st.get("chain_length", 1) * 104 / rate) * btc_price_usd,
            "network_health": "SOVEREIGN" if btc_status.get("status") == "SYNCHRONIZED" else "RESONATING",
            "timestamp": time.time()}


@router.get("/api/v1/mainnet/stream", tags=["Mainnet"])
async def mainnet_stream():
    async def event_generator():
        from l104_mainnet_bridge import mainnet_bridge
        from l104_sovereign_coin_engine import sovereign_coin
        from l104_sovereign_exchange import sovereign_exchange
        from l104_capital_offload import capital_offload
        while True:
            try:
                btc_status = mainnet_bridge.get_mainnet_status()
                coin_st = sovereign_coin.get_status()
                btc_price_usd = 100000.0
                try:
                    async with httpx.AsyncClient(timeout=3.0) as client:
                        resp = await client.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
                        if resp.status_code == 200:
                            btc_price_usd = resp.json().get("bitcoin", {}).get("usd", 100000.0)
                except Exception:
                    pass
                data = {"btc_balance": btc_status.get("confirmed_btc", 0),
                        "btc_pending": btc_status.get("unconfirmed_btc", 0),
                        "btc_price_usd": btc_price_usd,
                        "chain_height": coin_st.get("chain_length", 1),
                        "difficulty": coin_st.get("difficulty", 4),
                        "hashrate": coin_st.get("mining_stats", {}).get("hashrate", "0.00 H/s"),
                        "exchange_rate": sovereign_exchange.get_current_rate(),
                        "capital_sats": capital_offload.total_capital_generated_sats,
                        "network_status": "SYNCHRONIZED" if btc_status.get("status") == "SYNCHRONIZED" else "RESONATING",
                        "timestamp": time.time()}
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(5)
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                await asyncio.sleep(10)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/v1/mainnet/mine", tags=["Mainnet"])
async def mainnet_mine(address: str = "L104_GENESIS"):
    from l104_sovereign_coin_engine import sovereign_coin
    try:
        new_block = sovereign_coin.mine_block(address)
        if new_block:
            return {"status": "SUCCESS", "block_index": new_block.index,
                    "block_hash": new_block.hash[:32] + "...", "reward": 104,
                    "resonance": new_block.resonance, "timestamp": new_block.timestamp}
        return {"status": "MINING", "message": "Block mining in progress..."}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}


@router.get("/api/v1/mainnet/blocks", tags=["Mainnet"])
async def mainnet_blocks(limit: int = 10):
    from l104_sovereign_coin_engine import sovereign_coin
    blocks = sovereign_coin.chain[-limit:] if len(sovereign_coin.chain) >= limit else sovereign_coin.chain
    return {"blocks": [{"index": b.index, "hash": b.hash[:32] + "...",
                         "timestamp": b.timestamp, "tx_count": len(b.transactions),
                         "resonance": getattr(b, "resonance", 0.985)}
                        for b in reversed(blocks)],
            "total_blocks": len(sovereign_coin.chain)}


@router.get("/api/v1/mainnet/btc-network", tags=["Mainnet"])
async def btc_network_info():
    from l104_mainnet_bridge import mainnet_bridge
    return mainnet_bridge.get_btc_network_info()


@router.get("/api/v1/mainnet/btc-price", tags=["Mainnet"])
async def btc_price():
    from l104_mainnet_bridge import mainnet_bridge
    return {"btc_price_usd": mainnet_bridge.get_btc_price_usd(), "timestamp": time.time()}


@router.get("/api/v1/mainnet/transactions", tags=["Mainnet"])
async def mainnet_transactions(limit: int = 10):
    from l104_mainnet_bridge import mainnet_bridge
    return {"address": mainnet_bridge.address,
            "transactions": mainnet_bridge.get_address_transactions(limit)}


@router.get("/api/v1/mainnet/comprehensive", tags=["Mainnet"])
async def mainnet_comprehensive():
    from l104_mainnet_bridge import mainnet_bridge
    from l104_sovereign_coin_engine import sovereign_coin
    from l104_sovereign_exchange import sovereign_exchange
    from l104_capital_offload import capital_offload
    btc_status = mainnet_bridge.get_mainnet_status()
    btc_network = mainnet_bridge.get_btc_network_info()
    btc_price = mainnet_bridge.get_btc_price_usd()
    coin_st = sovereign_coin.get_status()
    rate = sovereign_exchange.get_current_rate()
    l104sp_btc_value = (coin_st.get("chain_length", 1) * 104 / rate)
    return {"l104sp": {"chain_height": coin_st.get("chain_length", 1),
                        "difficulty": coin_st.get("difficulty", 4),
                        "hashrate": coin_st.get("mining_stats", {}).get("hashrate", "0.00 H/s"),
                        "blocks_mined": coin_st.get("mining_stats", {}).get("blocks_mined", 0),
                        "total_supply_mined": coin_st.get("chain_length", 1) * 104},
            "btc_vault": {"address": btc_status.get("address"), "balance_btc": btc_status.get("confirmed_btc", 0),
                           "pending_btc": btc_status.get("unconfirmed_btc", 0),
                           "tx_count": btc_status.get("tx_count", 0), "sync_status": btc_status.get("status")},
            "btc_network": btc_network,
            "exchange": {"rate": rate, "l104sp_value_btc": l104sp_btc_value,
                          "l104sp_value_usd": l104sp_btc_value * btc_price, "btc_price_usd": btc_price},
            "capital": {"accumulated_sats": capital_offload.total_capital_generated_sats,
                         "transfers_count": len(capital_offload.transfer_log),
                         "connection_real": capital_offload.is_connection_real},
            "timestamp": time.time()}


# ─── SOCIAL AMPLIFIER ─────────────────────────────────────────────────────────

@router.get("/api/social/status", tags=["Social"])
async def social_status():
    from l104_social_amplifier import social_amplifier
    return social_amplifier.get_status()


@router.post("/api/social/add-target", tags=["Social"])
async def social_add_target(platform: str = "youtube", url: str = "", target_views: int = 10000):
    from l104_social_amplifier import social_amplifier
    target = social_amplifier.add_target(platform, url, target_views)
    return {"status": "SUCCESS", "platform": target.platform, "url": target.url,
            "target_views": target.target_views}


@router.get("/api/social/optimal-timing", tags=["Social"])
async def social_optimal_timing():
    from l104_social_amplifier import social_amplifier
    return social_amplifier.calculate_optimal_post_time()


@router.post("/api/social/content-seed", tags=["Social"])
async def social_content_seed(topic: str = "L104 Sovereign AI"):
    from l104_social_amplifier import social_amplifier
    return social_amplifier.generate_viral_content_seed(topic)


@router.get("/api/social/monetization", tags=["Social"])
async def social_monetization():
    from l104_social_amplifier import social_amplifier
    return social_amplifier.get_monetization_strategy()


@router.post("/api/social/amplify", tags=["Social"])
async def social_amplify(duration_minutes: int = 5):
    from l104_social_amplifier import social_amplifier
    return await social_amplifier.run_amplification_cycle(duration_minutes)


# ─── MINING CONTROL ───────────────────────────────────────────────────────────

@router.post("/api/mining/start", tags=["Mining"])
async def mining_start():
    import os
    import subprocess
    from pathlib import Path
    result = subprocess.run(["pgrep", "-f", "l104_fast_miner"], capture_output=True, text=True)
    if result.stdout.strip():
        return {"status": "ALREADY_RUNNING", "pids": result.stdout.strip().split("\n")}
    _base_dir = str(Path(__file__).parent.parent.absolute())
    proc = subprocess.Popen([".venv/bin/python", "l104_fast_miner.py"], cwd=_base_dir,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return {"status": "STARTED", "pid": proc.pid, "message": "L104 Fast Miner started in background"}


@router.post("/api/mining/stop", tags=["Mining"])
async def mining_stop():
    import subprocess
    subprocess.run(["pkill", "-f", "l104_fast_miner"], capture_output=True, text=True)
    return {"status": "STOPPED", "message": "Mining processes terminated"}


@router.get("/api/mining/stats", tags=["Mining"])
async def mining_stats():
    import subprocess
    from l104_sovereign_coin_engine import sovereign_coin
    result = subprocess.run(["pgrep", "-f", "l104_fast_miner"], capture_output=True, text=True)
    return {"miner_running": bool(result.stdout.strip()), "coin_status": sovereign_coin.get_status(),
            "difficulty": sovereign_coin.difficulty, "mining_reward": sovereign_coin.mining_reward,
            "chain_length": len(sovereign_coin.chain),
            "pending_transactions": len(sovereign_coin.pending_transactions)}


# ─── BITCOIN RESEARCH ─────────────────────────────────────────────────────────

@router.get("/api/v21/btc/report", tags=["Bitcoin Research"])
async def get_btc_research_report():
    from l104_bitcoin_researcher import L104BitcoinResearcher
    from fastapi.responses import PlainTextResponse
    researcher = L104BitcoinResearcher(target_difficulty_bits=28)
    return PlainTextResponse(researcher.bitcoin_derivation_report())


@router.post("/api/v21/btc/research", tags=["Bitcoin Research"])
async def start_btc_research_cycle(background_tasks: BackgroundTasks, iterations: int = 5000):
    from l104_bitcoin_researcher import L104BitcoinResearcher
    from config import BTC_ADDRESS
    researcher = L104BitcoinResearcher(target_difficulty_bits=28)
    if researcher.stop_event.is_set():
        researcher.stop_event.clear()
    background_tasks.add_task(researcher.run_parallel_search, iterations)
    return {"status": "Research Cycle Initiated", "policy": "DISCRETE", "iterations": iterations}


@router.get("/api/v21/btc/status", tags=["Bitcoin Research"])
async def get_btc_research_status():
    from l104_bitcoin_researcher import L104BitcoinResearcher
    from config import BTC_ADDRESS, L104_INVARIANT
    researcher = L104BitcoinResearcher(target_difficulty_bits=28)
    return {"address": BTC_ADDRESS, "hashes_performed": researcher.hashes_performed.value,
            "is_active": not researcher.stop_event.is_set(),
            "target": hex(researcher.target), "alignment": f"{L104_INVARIANT}"}
