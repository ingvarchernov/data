#!/usr/bin/env python3
import asyncio
from binance.client import Client
from config import BINANCE_CONFIG, TRADING_CONFIG
import logging
from datetime import datetime, timedelta
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def preflight_check():
    logger.info("="*100)
    logger.info("PREFLIGHT CHECK")
    logger.info("="*100)
    
    errors = []
    warnings = []
    
    logger.info("\n1. API Connection...")
    try:
        client = Client(
            BINANCE_CONFIG['api_key'],
            BINANCE_CONFIG['api_secret'],
            testnet=BINANCE_CONFIG['testnet'],
            requests_params={'timeout': 30}
        )
        
        account = await asyncio.get_event_loop().run_in_executor(None, client.futures_account)
        balance = float(account['totalWalletBalance'])
        logger.info(f"   OK Balance: ${balance:.2f}")
    except Exception as e:
        errors.append(f"API: {e}")
        logger.error(f"   ERROR {e}")
        return False
    
    logger.info("\n2. Open Positions...")
    try:
        positions = await asyncio.get_event_loop().run_in_executor(None, client.futures_position_information)
        
        open_pos = {}
        total_pnl = 0
        
        for pos in positions:
            amt = float(pos['positionAmt'])
            if abs(amt) > 0.0001:
                symbol = pos['symbol']
                pnl = float(pos['unRealizedProfit'])
                open_pos[symbol] = {'side': 'LONG' if amt > 0 else 'SHORT', 'entry': float(pos['entryPrice']), 'pnl': pnl}
                total_pnl += pnl
                
                if pnl < -25:
                    warnings.append(f"{symbol}: ${pnl:.2f}")
        
        logger.info(f"   Open: {len(open_pos)}/{TRADING_CONFIG['max_positions']} | PnL: ${total_pnl:+.2f}")
        
        if warnings:
            logger.warning(f"   WARNING Large losses: {', '.join(warnings[:3])}")
    except Exception as e:
        errors.append(f"Positions: {e}")
        logger.error(f"   ERROR {e}")
    
    logger.info("\n" + "="*100)
    
    if errors:
        logger.error(f"ERRORS: {len(errors)}")
        for e in errors:
            logger.error(f"   {e}")
        return False
    
    logger.info("OK READY TO START")
    return True

async def main():
    return await preflight_check()

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
