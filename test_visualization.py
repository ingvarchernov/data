#!/usr/bin/env python3
"""
Test Pattern Visualization - Generate sample charts
"""
import asyncio
import logging
from multi_timeframe_scanner import MultiTimeframeScanner
from pattern_chart_visualizer import visualize_top_patterns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_visualization():
    """Тестує візуалізацію паттернів"""
    
    logger.info("=" * 80)
    logger.info("TEST: Pattern Chart Visualization")
    logger.info("=" * 80)
    
    # Initialize scanner
    scanner = MultiTimeframeScanner()
    
    # Get top symbols
    logger.info("\n🔍 Getting top 20 symbols...")
    symbols = await scanner.get_top_symbols(limit=20)
    logger.info(f"Symbols: {', '.join(symbols[:10])}...")
    
    # Scan them
    logger.info("\n🔍 Scanning on 5 timeframes...")
    results = await scanner.batch_scan(symbols)
    
    # Print summary
    scanner.print_results(results, top_n=10)
    
    # Generate charts for top 5
    logger.info("\n" + "=" * 80)
    logger.info("📊 GENERATING CHARTS")
    logger.info("=" * 80)
    
    chart_files = await visualize_top_patterns(scanner, results, top_n=5)
    
    if chart_files:
        logger.info(f"\n✅ Generated {len(chart_files)} charts:")
        for filepath in chart_files:
            logger.info(f"  📈 {filepath}")
    else:
        logger.warning("\n⚠️ No charts generated (check matplotlib installation)")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Test completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_visualization())
