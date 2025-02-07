
         # Calculate Q-values for current state

        """Implements Peng's Q(λ) algorithm like the Java version"""
        q_values_current = self.calculate_all_q_values(state)
        best_action_current = self.find_action_with_max_value(q_values_current)
        max_qt = q_values_current[best_action_current]
        
        # Calculate max Q for previous state
        q_values_previous = self.calculate_all_q_values(self.last_state)
        best_action_previous = self.find_action_with_max_value(q_values_previous)
        max_q = q_values_previous[best_action_previous]
        
        for trace in self.traces:
            if trace.is_similar_to_state(state):
                if not trace.is_similar_to_action(action):
                    # Remove traces for same state but different actions
                    self.traces.remove(trace)
                else:
                    # Update trace value to 1 for matching state-action
                    trace.value = 1
                    
                    # Get current Q-value for this trace
                    qt = self.model(trace.state, trace.action)
                    
                    # Calculate new Q-value using Peng's Q(λ)
                    target_q = qt + self.alpha * trace.value * (reward + self.gamma * max_qt - max_q)
                    
                    # Train network
                    self.train_neural_network(trace.state, trace.action, target_q)
            else:
                # Decay trace value
                trace.value = self.gamma * self.lambda_param * trace.value
                
                # Update Q-value for this trace
                qt = self.model(trace.state, trace.action)
                target_q = qt + self.alpha * trace.value * (reward + self.gamma * max_qt - max_q)
                
                # Train network
                self.train_neural_network(trace.state, trace.action, target_q)
