    async def step(self, observation: Dict[str, Any], oracle_output: Dict[str, Any]) -> Tuple[int, float]:
        """
        Main decision step.
        """
        # 1. Confidence Veto
        if oracle_output.get('status') == "INSUFFICIENT_DATA":
            print("Strategist: Holding due to Low Oracle Confidence.")
            return 0, 0.0 # Force HOLD
        
        # 2. Evaluate
        action, parameter = self.evaluate_action_space(observation)
        
        # 3. Shadow Mode Logic
        if self.system_mode == 'SHADOW' and action != 0:
            import asyncio
            import random
            from datetime import datetime
            
            # Simulate "Virtual Processing" delay
            delay = random.uniform(5.0, 15.0)
            self.logger.info(f"Shadow Mode: Simulating Execution Delay of {delay:.2f}s...")
            await asyncio.sleep(delay)
            
            # Log Virtual Fill
            if self.db:
                # Assuming DataBridge has a method or we execute raw SQL
                # For now, we mock the record structure
                record = {
                    "time": datetime.utcnow(),
                    "action_type": action,
                    "type_id": observation.get('type_id', 0),
                    "price": observation.get('price', 0), # Or calculated fill price
                    "quantity": 100, # Default lot
                    "simulated": True,
                    "status": "VIRTUAL_FILL"
                }
                # Using the bridge to push (Needs specific method or generic table push)
                await self.db.push_to_postgres("trade_logs", [record])
                
            self.logger.info(f"Shadow Mode: Virtual Fill Logged. Action: {action} @ {parameter}")

        return action, parameter