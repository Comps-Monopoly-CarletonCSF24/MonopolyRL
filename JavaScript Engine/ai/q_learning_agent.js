//import { Player } from './player.js';
//import { GameSettings } from './settings.js';
import { State } from "./state.js";
import { chooseAction  } from "./run_onnx.js";
// Whether the trade function is allowed
var ToggleTrade = false;

// The purpose of this AI is not to be a relistic opponant, but to give an example of a vaild AI player.
// This is an implementation of the fixed agent
// The p is player
export async function QLearning(p) {
    this.alertList = "";
    // This variable is static, it is not related to each instance.
    this.constructor.count++;

    p.name = "QLearning " + this.constructor.count; // this gets your ai a proper 

    // Decide whether to buy a property the AI landed on.
    // Return: boolean (true to buy).
    // Arguments:
    // index: the property's index (0-39)
    this.buyProperty = function(index) {
        console.log("buyProperty");
        var s = square[index]; // get the value of the square at the given index

        if (p.money > s.price+10) {
            return true;
        } else {
            return false;
        }

    }

    // Determine the response to an offered trade.
    // Return: boolean/instanceof Trade: a valid Trade object to counter offer (with the AI as the recipient); false to decline; true to accept.
    // Arguments:
    // tradeObj: the proposed trade, an instanceof Trade, has the AI as the recipient.
    this.acceptTrade = function(tradeObj) {
        console.log("acceptTrade");

        var tradeValue = 0;
        var money = tradeObj.getMoney();   // money offered in the trade
        var initiator = tradeObj.getInitiator(); // the person offering to trade
        var recipient = tradeObj.getRecipient(); // the person receiving the trade offer (I assume it would be this ai)
        var property = [];

        // increase trade value by 10 depending on whether the offer is an out-of-jail card
        tradeValue += 10 * tradeObj.getCommunityChestJailCard();
        tradeValue += 10 * tradeObj.getChanceJailCard();

        // I am thinking this is the case the person is offering money on top of the jail card or if the jail card even was an option in the first place
        tradeValue += money;  

        // creates a new property similar to the one offered in trade
        // creates trade_value by getting the property's price and halving the price if the property is mortgaged. 
        for (var i = 0; i < 40; i++) { // why do you need to go through all the properties?
            property[i] = tradeObj.getProperty(i);
            tradeValue += tradeObj.getProperty(i) * square[i].price * (square[i].mortgage ? 0.5 : 1);
        }

        console.log(tradeValue);

        var proposedMoney = 15 - tradeValue + money; // trying to make 25 bucks off the trade. Will be useful in request

        // By any property that's offering you more than $25 buck??? Insane
        if (tradeValue > 15) {
            return true;
        // If they are requesting more than $50 or offering you less than the 25 you wanted to save,
        // offer them a new trade that involves the same property and the 25 bucks
        } else if (tradeValue >= -50 && initiator.money > proposedMoney) {

            return new Trade(initiator, recipient, proposedMoney, property, tradeObj.getCommunityChestJailCard(), tradeObj.getChanceJailCard());
        }

        return false;
    }

    // This function is called at the beginning of the AI's turn, before any dice are rolled. 
    // The purpose is to allow the AI to manage property and/or initiate trades. (does every player get the same privileges?)
    // Return: boolean: Must return true if and only if the AI proposed a trade. (does it participate in other trades too?)
    this.beforeTurn = function() {
        console.log("beforeTurn");
        console.log(chooseAction())
        var state = new State()
        console.log(state.state)
        var s;
        var allGroupOwned;
        var max;
        var leastHouseProperty;
        var leastHouseNumber;

        // Buy houses.
        for (var i = 0; i < 40; i++) {
            s = square[i];

            if (s.owner === p.index && s.groupNumber >= 3) {
                max = s.group.length;
                allGroupOwned = true;
                leastHouseNumber = 6; // No property will ever have 6 houses.

                for (var j = max - 1; j >= 0; j--) {
                    if (square[s.group[j]].owner !== p.index) {
                        allGroupOwned = false;
                        break;
                    }

                    if (square[s.group[j]].house < leastHouseNumber) {
                        leastHouseProperty = square[s.group[j]];
                        leastHouseNumber = leastHouseProperty.house;
                    }
                }

                if (!allGroupOwned) {
                    continue;
                }

                if (p.money > leastHouseProperty.houseprice + 100) {
                    buyHouse(leastHouseProperty.index);
                }


            }
        }

        // Unmortgage property
        for (var i = 39; i >= 0; i--) {
            s = square[i];

            if (s.owner === p.index && s.mortgage && p.money > s.price) {
                unmortgage(i);
            }
        }

        return false;
    }

    var utilityForRailroadFlag = true; // Don't offer this trade more than once.


    // This function is called every time the AI lands on a square. 
    // The purpose is to allow the AI to manage property and/or initiate trades.
    // Return: boolean: Must return true if and only if the AI proposed a trade.
    this.onLand = function() {
        console.log("onLand");
        var proposedTrade;
        var property = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        var railroadIndexes = [5, 15, 25, 35]; // Why do we need railroads index? because it will come in handy when we have only one utility. Read below 
        var requestedRailroad;
        var offeredUtility;
        var s;

        // If AI owns exactly one utility, try to trade it for a railroad. (why is that ideal? why would you want to do that? Are railroads the most pricey things?)
        // get the railroad index
        for (var i = 0; i < 4; i++) {
            s = square[railroadIndexes[i]];

            if (s.owner !== 0 && s.owner !== p.index) {
                requestedRailroad = s.index;
                break;
            }
        }
        
        // get the utility to trade for the railroad
        if (square[12].owner === p.index && square[28].owner !== p.index) {
            offeredUtility = 12;
        } else if (square[28].owner === p.index && square[12].owner !== p.index) {
            offeredUtility = 28;
        }

        // not offered the trade before and the 2 dice values are different and the trade params are available
        if (utilityForRailroadFlag && game.getDie(1) !== game.getDie(2) && requestedRailroad && offeredUtility) {
        
        // Propose trade
        if (ToggleTrade && utilityForRailroadFlag && game.getDie(1) !== game.getDie(2) && requestedRailroad && offeredUtility) {
            utilityForRailroadFlag = false;
            property[requestedRailroad] = -1;
            property[offeredUtility] = 1;

            proposedTrade = new Trade(p, player[square[requestedRailroad].owner], 0, property, 0, 0)

            game.trade(proposedTrade);
            return true;
        }

        return false;
    }

    // Determine whether to post bail/use get out of jail free card (if in possession).
    // Return: boolean: true to post bail/use card.
    this.postBail = function() {
        console.log("postBail");

        // p.jailroll === 2 on third turn in jail. 
        // Only try to use your jailcard or pay the out of jail money only if you are in jail for the third roll (you were not able to hit doubles on any of your previous rolls)
        if ((p.communityChestJailCard || p.chanceJailCard) && p.jailroll === 1) {
            return true;
        } else {
            return false;
        }
    }

    // Mortgage enough properties to pay debt.
    // Return: void: don't return anything, just call the functions mortgage()/sellhouse()
    // This function just morgages a house in case we need to and make sure we get some money by doing that action
    this.payDebt = function() {
        console.log("payDebt");
        for (var i = 39; i >= 0; i--) {
            s = square[i];

            if (s.owner === p.index && !s.mortgage && s.house === 0) {
                mortgage(i);
                console.log(s.name);
            }

            if (p.money >= 0) {
                return;
            }
        }

    }

    // Determine what to bid during an auction.
    // Return: integer: -1 for exit auction, 0 for pass, a positive value for the bid.
    this.bid = function(property, currentBid) {
        console.log("bid");
        var bid;
        // max bid is 20 + 10 = 30
        bid = currentBid + Math.round(Math.random() * 20 + 10);	
        // if the machine cannot afford the bid, i.e it has less than bid + 50 or the bid it came up with is propertyPrice + 1/2 property price, then exit bid. 
        if (p.money < bid + 50 || bid > square[property].price * 1.5) {
            return -1;
        } else {
            return bid;
        }

    }
    }
}

/*
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
}*/