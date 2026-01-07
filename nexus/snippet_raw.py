
# --- Raw Data Stream (Librarian Pulse) ---
st.header("Raw Data Stream (Librarian Pulse)")
try:
    # Use environment variable for DB connection inside Docker
    db_conn_str = os.getenv("DATABASE_URL", "postgresql://eve_user:eve_password@db:5432/eve_market_data")
    engine = sqlalchemy.create_engine(db_conn_str)
    
    # Query Last 20 rows from market_history
    # If using order streaming, we might check market_orders instead?
    # User asked for market_history table specifically in prompt, but Librarian streams to market_orders now.
    # I will query market_orders as that shows the heartbeat best.
    
    with engine.connect() as conn:
        # Check market_orders for activity
        query = text("""
            SELECT timestamp, type_id, price, volume_remaining, is_buy_order, issue_date 
            FROM market_orders 
            ORDER BY timestamp DESC 
            LIMIT 20
        """)
        df_raw = pd.read_sql(query, conn)
        
    st.dataframe(df_raw, use_container_width=True)
    
except Exception as e:
    st.error(f"Waiting for Data Stream... ({str(e)})")
