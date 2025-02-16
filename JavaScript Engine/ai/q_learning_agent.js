import { Player } from './player.js';
import { State } from './model.js';
import { GameSettings } from './settings.js';

export class DQAPlayer extends Player {
    constructor(name, settings) {
        super(name, settings);
        this.action = { actionType: "do_nothing" };
        this.actionSuccessful = false;
        this.model = null; // to be loaded later 
    }

    async loadModel() {
        this.model = await tf.loadLayersModel('models/model.json');
    }

    async handleAction(board, players, dice, log) {
        for (let groupIdx = 0; groupIdx < board.groupCellIndices.length; groupIdx++) {
            if (this.isGroupActionable(groupIdx, board)) {
                const [state, action] = await this.selectAction(players);
                this.actionSuccessful = this.executeAction(
                    board, players, log, action, groupIdx
                );
            }
        }
    }

    async selectAction(players) {
        const currentState = new State(this, players);
        const currentAction = await this.chooseAction(currentState);
        return [currentState, currentAction];
    }

    async chooseAction(state) {
        const allActions = new Array(84).fill(0).map((_, i) => ({
            actionIndex: i,
            actionType: this.mapActionIndexToType(i)
        }));

        let bestAction = null;
        let maxQValue = -Infinity;

        for (const action of allActions) {
            const input = tf.tensor([...state.state, action.actionIndex]);
            const qValue = await this.model.predict(input).data();
            if (qValue[0] > maxQValue) {
                maxQValue = qValue[0];
                bestAction = action;
            }
        }

        return bestAction;
    }

    mapActionIndexToType(actionIndex) {
        const actions = ['buy', 'sell', 'do_nothing'];
        return actions[actionIndex % actions.length];
    }

    executeAction(board, players, log, action, groupIdx) {
        switch (action.actionType) {
            case 'buy':
                return this.buyInGroup(groupIdx, board, players, log);
            case 'sell':
                return this.sellInGroup(groupIdx, board, log);
            case 'do_nothing':
                return true;
            default:
                return false;
        }
    }

    buyInGroup(groupIdx, board, players, log) {
        const cellsInGroup = board.groupCellIndices[groupIdx].map(idx => board.cells[idx]);

        const getNextPropertyToUnmortgage = () => {
            for (const cell of cellsInGroup) {
                if (!cell.isMortgaged) continue;
                if (cell.owner !== this) continue;

                const costToUnmortgage = 
                    cell.costBase * GameSettings.mortgageValue +
                    cell.costBase * GameSettings.mortgageFee;

                if (this.money - costToUnmortgage < this.settings.unspendableCash) continue;

                return [cell, costToUnmortgage];
            }
            return [null, null];
        };

        const unmortgageProperty = (propertyToUnmortgage, costToUnmortgage) => {
            log.add(`${this.name} unmortgages ${propertyToUnmortgage.name} for $${costToUnmortgage}`);
            this.money -= costToUnmortgage;
            propertyToUnmortgage.isMortgaged = false;
            this.updateListsOfPropertiesToTrade(board);
            return true;
        };

        const canBuyProperty = () => {
            const propertyToBuy = board.cells[this.position];
            if (!board.groupCellIndices[groupIdx].includes(this.position)) return false;
            if (!propertyToBuy.isProperty) return false;
            if (propertyToBuy.owner !== null) return false;
            if (this.money - propertyToBuy.costBase < this.settings.unspendableCash) return false;
            return true;
        };

        const buyProperty = () => {
            const propertyToBuy = board.cells[this.position];
            propertyToBuy.owner = this;
            this.owned.push(propertyToBuy);
            this.money -= propertyToBuy.costBase;
            log.add(`Player ${this.name} bought ${propertyToBuy.name} for $${propertyToBuy.costBase}`);
            
            board.recalculateMonopolyCoeffs(propertyToBuy);
            players.forEach(player => player.updateListsOfPropertiesToTrade(board));
            return true;
        };

        const getNextPropertyToImprove = () => {
            const canBeImproved = [];
            for (const cell of cellsInGroup) {
                if (cell.owner !== this) return null;
                if (cell.hasHotel === 0 && !cell.isMortgaged && cell.monopolyCoef === 2 &&
                    cell.group !== "Railroads" && cell.group !== "Utilities") {
                    
                    // Check other cells in group
                    const otherCellsOk = board.groups[cell.group].every(otherCell => 
                        otherCell.hasHouses >= cell.hasHouses && !otherCell.isMortgaged
                    );

                    if (otherCellsOk) {
                        if ((cell.hasHouses !== 4 && board.availableHouses > 0) ||
                            (cell.hasHouses === 4 && board.availableHotels > 0)) {
                            canBeImproved.push(cell);
                        }
                    }
                }
            }
            return canBeImproved.sort((a, b) => a.costHouse - b.costHouse)[0] || null;
        };

        // Helper function to improve property
        const improveProperty = (cellToImprove) => {
            if (!cellToImprove) return false;
            const improvementCost = cellToImprove.costHouse;
            if (this.money - improvementCost < this.settings.unspendableCash) return false;

            const ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4:"4th"};
            
            if (cellToImprove.hasHouses !== 4) {
                cellToImprove.hasHouses++;
                board.availableHouses--;
                this.money -= cellToImprove.costHouse;
                log.add(`${this.name} built ${ordinal[cellToImprove.hasHouses]} house on ${cellToImprove.name} for $${cellToImprove.costHouse}`);
            } else {
                cellToImprove.hasHouses = 0;
                cellToImprove.hasHotel = 1;
                board.availableHouses += 4;
                board.availableHotels--;
                this.money -= cellToImprove.costHouse;
                log.add(`${this.name} built a hotel on ${cellToImprove.name}`);
            }
            return true;
        };

        // Execute buying strategy
        const cellToImprove = getNextPropertyToImprove();
        if (cellToImprove) {
            return improveProperty(cellToImprove);
        }

        const [cellToUnmortgage, costToUnmortgage] = getNextPropertyToUnmortgage();
        if (cellToUnmortgage) {
            return unmortgageProperty(cellToUnmortgage, costToUnmortgage);
        }

        if (canBuyProperty()) {
            return buyProperty();
        }

        return false;
    }

    sellInGroup(groupIdx, board, log) {
        const cellsInGroup = board.groupCellIndices[groupIdx].map(idx => board.cells[idx]);

        // Helper function to get next property to sell
        const getNextPropertyToSell = () => {
            for (const cell of cellsInGroup) {
                if (!cell.isProperty) continue;
                if (cell.owner !== this) continue;
                if (cell.hasHouses > 0 || cell.hasHotel > 0) continue;
                return cell;
            }
            return null;
        };

        // for mortgaging
        const mortgageProperty = (propertyToSell) => {
            const mortgagePrice = propertyToSell.costBase * GameSettings.mortgageValue;
            this.money += mortgagePrice;
            log.add(`${this.name} mortgages ${propertyToSell.name}, raising $${mortgagePrice}`);
            return true;
        };

        const getNextPropertyToDowngrade = () => {
            const canBeDowngraded = [];
            for (const cell of cellsInGroup) {
                if (cell.owner === this && (cell.hasHotel === 1 || cell.hasHouses > 0)) {
                    const otherCellsOk = board.groups[cell.group].every(otherCell => 
                        otherCell.hasHouses <= cell.hasHouses
                    );
                    if (otherCellsOk) {
                        canBeDowngraded.push(cell);
                    }
                }
            }
            return canBeDowngraded
                .sort((a, b) => (b.hasHotel * 5 + b.hasHouses) - (a.hasHotel * 5 + a.hasHouses))[0] || null;
        };

        // actually downgrade it
        const downgradeProperty = (cellToDowngrade) => {
            if (!cellToDowngrade) return false;

            if (cellToDowngrade.hasHotel === 1) {
                if (board.availableHouses >= 4) {
                    cellToDowngrade.hasHotel = 0;
                    cellToDowngrade.hasHouses = 4;
                    board.availableHotels++;
                    board.availableHouses -= 4;
                    const sellPrice = Math.floor(cellToDowngrade.costHouse / 2);
                    this.money += sellPrice;
                    log.add(`${this.name} downgraded hotel to houses on ${cellToDowngrade.name} for $${sellPrice}`);
                    return true;
                }
            } else if (cellToDowngrade.hasHouses > 0) {
                cellToDowngrade.hasHouses--;
                board.availableHouses++;
                const sellPrice = Math.floor(cellToDowngrade.costHouse / 2);
                this.money += sellPrice;
                log.add(`${this.name} sold house on ${cellToDowngrade.name} for $${sellPrice}`);
                return true;
            }
            return false;
        };

        let cellToDowngrade = getNextPropertyToDowngrade();
        if (cellToDowngrade) {
            return downgradeProperty(cellToDowngrade);
        }

        const cellToSell = getNextPropertyToSell();
        if (cellToSell) {
            return mortgageProperty(cellToSell);
        }

        cellToDowngrade = getNextPropertyToDowngrade();
        if (cellToDowngrade) {
            return downgradeProperty(cellToDowngrade);
        }

        return false;
    }

    isGroupActionable(groupIdx, board) {
        const cellIndicesInGroup = board.groupCellIndices[groupIdx];
        return cellIndicesInGroup.some(cellIdx => {
            const cell = board.cells[cellIdx];
            return (
                cell.owner === this || // can sell
                (this.position === cellIdx && !cell.owner) || // can buy
                cell.monopolyCoef === 2 // can improve
            );
        });
    }
}