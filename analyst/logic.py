def filter_market_opportunities(candidates, min_spread=0.005, min_volume=1000):
    filtered = []
    for item in candidates:
        spread = item.get('spread', 0)
        volume = item.get('volume', 0)
        if spread >= min_spread and volume >= min_volume:
            filtered.append(item)
    return filtered

def calculate_liquidity_depth(order_book, side='ask', slippage_tolerance=0.002):
    if not order_book: return 0
    sorted_orders = sorted(order_book, key=lambda x: x['price'], reverse=(side == 'bid'))
    best_price = sorted_orders[0]['price']
    
    if side == 'ask':
        limit_price = best_price * (1 + slippage_tolerance)
        valid_orders = [o for o in sorted_orders if o['price'] <= limit_price]
    else:
        limit_price = best_price * (1 - slippage_tolerance)
        valid_orders = [o for o in sorted_orders if o['price'] >= limit_price]
        
    max_volume = sum(o['volume_remain'] for o in valid_orders)
    return max_volume

def calculate_order_book_walls(order_book, side='ask', depth_levels=10):
    if not order_book:
        return {'total_volume': 0, 'wall_price_limit': 0, 'levels_counted': 0}
    price_map = {}
    for o in order_book:
        p = o['price']
        price_map[p] = price_map.get(p, 0) + o['volume_remain']
    sorted_prices = sorted(price_map.keys(), reverse=(side == 'bid'))
    top_prices = sorted_prices[:depth_levels]
    total_volume = sum(price_map[p] for p in top_prices)
    wall_limit = top_prices[-1] if top_prices else 0
    return {
        'total_volume': total_volume,
        'wall_price_limit': wall_limit,
        'levels_counted': len(top_prices)
    }
