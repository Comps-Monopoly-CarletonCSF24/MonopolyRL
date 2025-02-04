class Action {
    constructor() {
        this.properties = Array.from({ length: 28 }, (_, i) => i); // Property indices from 0 to 27
        this.actions = ['buy', 'sell', 'do_nothing']; // Available actions for each property
        this.totalActions = this.properties.length * this.actions.length; // 1x84 action space
    }

    mapActionIndex(actionIndex) {
        /**
         * Maps a flattened action index to a specific property and action type.
         *
         * @param {number} actionIndex - The index of the action in the flattened action space.
         * @returns {[number, string]} A tuple containing the property index and the action type.
         * @throws {Error} If actionIndex is not a valid integer within range.
         */
        if (!Number.isInteger(actionIndex)) {
            throw new Error(`actionIndex must be an integer, got ${typeof actionIndex}.`);
        }
        if (actionIndex < 0 || actionIndex >= this.totalActions) {
            throw new Error(`The action index must be between 0 and ${this.totalActions - 1}.`);
        }
        const propertyIdx = Math.floor(actionIndex / this.actions.length);
        const actionType = this.actions[actionIndex % this.actions.length];
        return [propertyIdx, actionType];
    }

    isExecutable(player, board, propertyIdx, actionIdx) {
        /**
         * Checks if the player can take the action they are attempting.
         * @param {Player} player - The player taking the action.
         * @param {Board} board - The game board.
         * @param {number} propertyIdx - The index of the property.
         * @param {number} actionIdx - The index of the action.
         * @returns {boolean} True if the action can be executed, false otherwise.
         */
        if (actionIdx === 1) {
            return true;
        } else if (actionIdx === 0) {
            const property = board[propertyIdx];
            return !property.isOwned() && player.canAfford(property.price);
        }
        return false;
    }

    executeAction(player, board, propertyIdx, actionType) {
        /**
         * Executes the action on the given property for the specified player.
         * @param {Player} player - The player taking the action.
         * @param {Board} board - The game board.
         * @param {number} propertyIdx - The index of the property.
         * @param {string} actionType - The type of action to execute.
         */
        const property = board[propertyIdx];
        if (actionType === 'buy') {
            if (!property.isOwned() && player.canAfford(property.price)) {
                player.buyProperty(property);
            }
        } else if (actionType === 'sell') {
            if (property.owner === player) {
                player.sellProperty(property);
            }
        }
    }
}
