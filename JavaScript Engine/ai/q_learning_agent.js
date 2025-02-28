// //import { Player } from './player.js';
// //import { GameSettings } from './settings.js';
// import { State } from "./state.js";
// import { chooseAction  } from "./run_onnx.js";
// // Whether the trade function is allowed
// var ToggleTrade = false;


// // The purpose of this AI is not to be a relistic opponant, but to give an example of a vaild AI player.
// // This is an implementation of the fixed agent
// // The p is player
// export class QLearning {
//     constructor(p) {
//         console.log("constructor called");
//         return new Promise(async (resolve, reject) => {
//             this.alertList = "";
//             // This variable is static, it is not related to each instance.
//             this.constructor.count++;

//             p.name = "QLearning " + this.constructor.count; // this gets your ai a proper 

//             // Decide whether to buy a property the AI landed on.
//             // Return: boolean (true to buy).
//             // Arguments:
//             // index: the property's index (0-39)
//             this.buyProperty = function(index) {
//                 console.log("buyProperty");
//                 var s = square[index]; // get the value of the square at the given index

//                 if (p.money > s.price+10) {
//                     return true;
//                 } else {
//                     return false;
//                 }

//             }

//             // Determine the response to an offered trade.
//             // Return: boolean/instanceof Trade: a valid Trade object to counter offer (with the AI as the recipient); false to decline; true to accept.
//             // Arguments:
//             // tradeObj: the proposed trade, an instanceof Trade, has the AI as the recipient.
//             this.acceptTrade = function(tradeObj) {
//                 console.log("acceptTrade");

//                 var tradeValue = 0;
//                 var money = tradeObj.getMoney();   // money offered in the trade
//                 var initiator = tradeObj.getInitiator(); // the person offering to trade
//                 var recipient = tradeObj.getRecipient(); // the person receiving the trade offer (I assume it would be this ai)
//                 var property = [];

//                 // increase trade value by 10 depending on whether the offer is an out-of-jail card
//                 tradeValue += 10 * tradeObj.getCommunityChestJailCard();
//                 tradeValue += 10 * tradeObj.getChanceJailCard();

//                 // I am thinking this is the case the person is offering money on top of the jail card or if the jail card even was an option in the first place
//                 tradeValue += money;  

//                 // creates a new property similar to the one offered in trade
//                 // creates trade_value by getting the property's price and halving the price if the property is mortgaged. 
//                 for (var i = 0; i < 40; i++) { // why do you need to go through all the properties?
//                     property[i] = tradeObj.getProperty(i);
//                     tradeValue += tradeObj.getProperty(i) * square[i].price * (square[i].mortgage ? 0.5 : 1);
//                 }

//                 console.log(tradeValue);

//                 var proposedMoney = 15 - tradeValue + money; // trying to make 25 bucks off the trade. Will be useful in request

//                 // By any property that's offering you more than $25 buck??? Insane
//                 if (tradeValue > 15) {
//                     return true;
//                 // If they are requesting more than $50 or offering you less than the 25 you wanted to save,
//                 // offer them a new trade that involves the same property and the 25 bucks
//                 } else if (tradeValue >= -50 && initiator.money > proposedMoney) {

//                     return new Trade(initiator, recipient, proposedMoney, property, tradeObj.getCommunityChestJailCard(), tradeObj.getChanceJailCard());
//                 }

//                 return false;
//             }

//             // This function is called at the beginning of the AI's turn, before any dice are rolled. 
//             // The purpose is to allow the AI to manage property and/or initiate trades. (does every player get the same privileges?)
//             // Return: boolean: Must return true if and only if the AI proposed a trade. (does it participate in other trades too?)
//             this.beforeTurn = async function() {
//                 console.log("beforeTurn");
//                 console.log(await chooseAction())
//                 var state = new State()
//                 var s;
//                 var allGroupOwned;
//                 var max;
//                 var leastHouseProperty;
//                 var leastHouseNumber;

//                 // Buy houses.
//                 for (var i = 0; i < 40; i++) {
//                     s = square[i];

//                     if (s.owner === p.index && s.groupNumber >= 3) {
//                         max = s.group.length;
//                         allGroupOwned = true;
//                         leastHouseNumber = 6; // No property will ever have 6 houses.

//                         for (var j = max - 1; j >= 0; j--) {
//                             if (square[s.group[j]].owner !== p.index) {
//                                 allGroupOwned = false;
//                                 break;
//                             }

//                             if (square[s.group[j]].house < leastHouseNumber) {
//                                 leastHouseProperty = square[s.group[j]];
//                                 leastHouseNumber = leastHouseProperty.house;
//                             }
//                         }

//                         if (!allGroupOwned) {
//                             continue;
//                         }

//                         if (p.money > leastHouseProperty.houseprice + 100) {
//                             buyHouse(leastHouseProperty.index);
//                         }


//                     }
//                 }

//                 // Unmortgage property
//                 for (var i = 39; i >= 0; i--) {
//                     s = square[i];

//                     if (s.owner === p.index && s.mortgage && p.money > s.price) {
//                         unmortgage(i);
//                     }
//                 }

//                 return false;
//             }

//             var utilityForRailroadFlag = true; // Don't offer this trade more than once.


//             // This function is called every time the AI lands on a square. 
//             // The purpose is to allow the AI to manage property and/or initiate trades.
//             // Return: boolean: Must return true if and only if the AI proposed a trade.
//             this.onLand = function() {
//                 console.log("onLand");
//                 var proposedTrade;
//                 var property = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
//                 var railroadIndexes = [5, 15, 25, 35]; // Why do we need railroads index? because it will come in handy when we have only one utility. Read below 
//                 var requestedRailroad;
//                 var offeredUtility;
//                 var s;

//                 // If AI owns exactly one utility, try to trade it for a railroad. (why is that ideal? why would you want to do that? Are railroads the most pricey things?)
//                 // get the railroad index
//                 for (var i = 0; i < 4; i++) {
//                     s = square[railroadIndexes[i]];

//                     if (s.owner !== 0 && s.owner !== p.index) {
//                         requestedRailroad = s.index;
//                         break;
//                     }
//                 }
                
//                 // get the utility to trade for the railroad
//                 if (square[12].owner === p.index && square[28].owner !== p.index) {
//                     offeredUtility = 12;
//                 } else if (square[28].owner === p.index && square[12].owner !== p.index) {
//                     offeredUtility = 28;
//                 }

//                 // not offered the trade before and the 2 dice values are different and the trade params are available
//                 if (utilityForRailroadFlag && game.getDie(1) !== game.getDie(2) && requestedRailroad && offeredUtility) {
                
//                 // Propose trade
//                 if (ToggleTrade && utilityForRailroadFlag && game.getDie(1) !== game.getDie(2) && requestedRailroad && offeredUtility) {
//                     utilityForRailroadFlag = false;
//                     property[requestedRailroad] = -1;
//                     property[offeredUtility] = 1;

//                     proposedTrade = new Trade(p, player[square[requestedRailroad].owner], 0, property, 0, 0)

//                     game.trade(proposedTrade);
//                     return true;
//                 }

//                 return false;
//             }

//             // Determine whether to post bail/use get out of jail free card (if in possession).
//             // Return: boolean: true to post bail/use card.
//             this.postBail = function() {
//                 console.log("postBail");

//                 // p.jailroll === 2 on third turn in jail. 
//                 // Only try to use your jailcard or pay the out of jail money only if you are in jail for the third roll (you were not able to hit doubles on any of your previous rolls)
//                 if ((p.communityChestJailCard || p.chanceJailCard) && p.jailroll === 1) {
//                     return true;
//                 } else {
//                     return false;
//                 }
//             }

//             // Mortgage enough properties to pay debt.
//             // Return: void: don't return anything, just call the functions mortgage()/sellhouse()
//             // This function just morgages a house in case we need to and make sure we get some money by doing that action
//             this.payDebt = function() {
//                 console.log("payDebt");
//                 for (var i = 39; i >= 0; i--) {
//                     s = square[i];

//                     if (s.owner === p.index && !s.mortgage && s.house === 0) {
//                         mortgage(i);
//                         console.log(s.name);
//                     }

//                     if (p.money >= 0) {
//                         return;
//                     }
//                 }

//             }

//             // Determine what to bid during an auction.
//             // Return: integer: -1 for exit auction, 0 for pass, a positive value for the bid.
//             this.bid = function(property, currentBid) {
//                 console.log("bid");
//                 var bid;
//                 // max bid is 20 + 10 = 30
//                 bid = currentBid + Math.round(Math.random() * 20 + 10);	
//                 // if the machine cannot afford the bid, i.e it has less than bid + 50 or the bid it came up with is propertyPrice + 1/2 property price, then exit bid. 
//                 if (p.money < bid + 50 || bid > square[property].price * 1.5) {
//                     return -1;
//                 } else {
//                     return bid;
//                 }

//             }
//             }
//             // Resolve the constructor
//             resolve(this);
//         })
//     }
// }

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

import { State } from "./state.js";
import { chooseAction } from "./run_onnx.js";
import { Actions } from "./action.js";
// Whether the trade function is allowed
var ToggleTrade = false;

export class QLearning {
    constructor(p) {
        console.log("constructor called");
        return new Promise(async (resolve, reject) => {
            this.alertList = "";
            // This variable is static, it is not related to each instance.
            if (!this.constructor.count) this.constructor.count = 0;
            this.constructor.count++;

            p.name = "QLearning " + this.constructor.count;
            
            // Add a property to track if last action was successful
            this.actionSuccessful = false;

            // Get property groups - similar to Python group structure
            this.getPropertyGroups = function() {
                const groups = {};
                
                for (let i = 0; i < 40; i++) {
                    const s = square[i];
                    if (s.groupNumber) {
                        if (!groups[s.groupNumber]) {
                            groups[s.groupNumber] = {
                                cells: [],
                                ownedByPlayer: 0,
                                totalProperties: 0
                            };
                        }
                        
                        groups[s.groupNumber].cells.push(i);
                        groups[s.groupNumber].totalProperties++;
                        
                        if (s.owner === p.index) {
                            groups[s.groupNumber].ownedByPlayer++;
                        }
                    }
                }
                
                return groups;
            };
            
            // Check if a group is actionable (similar to Python's is_group_actionable)
            this.isGroupActionable = function(groupIdx) {
                const cellIndicesInGroup = [];
                
                // Get all cells in this group
                for (let i = 0; i < 40; i++) {
                    if (square[i].groupNumber === parseInt(groupIdx)) {
                        cellIndicesInGroup.push(i);
                    }
                }
                
                for (const cellIdx of cellIndicesInGroup) {
                    const cell = square[cellIdx];
                    
                    // Can sell if player owns property
                    if (cell.owner === p.index) {
                        return true;
                    }
                    
                    // Can buy if player is on property and it's unowned
                    if (p.position === cellIdx && !cell.owner) {
                        return true;
                    }
                    
                    // Can improve if monopoly
                    if (this.isMonopoly(groupIdx)) {
                        return true;
                    }
                }
                
                return false;
            };
            
            // Check if player has monopoly on a group
            this.isMonopoly = function(groupIdx) {
                const cellsInGroup = [];
                let ownedByPlayer = 0;
                let totalInGroup = 0;
                
                for (let i = 0; i < 40; i++) {
                    if (square[i].groupNumber === parseInt(groupIdx)) {
                        cellsInGroup.push(square[i]);
                        totalInGroup++;
                        
                        if (square[i].owner === p.index) {
                            ownedByPlayer++;
                        }
                    }
                }
                
                return ownedByPlayer === totalInGroup && totalInGroup > 0;
            };

            // Decide whether to buy a property the AI landed on.
            this.buyProperty = async function(index) {
                console.log("buyProperty function called for index:", index);
                var s = square[index];
                
                // Get model's recommendation
                const action = await chooseAction();
                console.log("AI action recommendation:", action);
                
                if (action === "buy") {
                    return p.money > s.price;
                } else if (action === "do_nothing") {
                    return false;
                }
                
                // Fallback strategy if model doesn't have a clear preference
                return p.money > s.price + 50;
            }

            // Handle trade offers - keeping original logic
            this.acceptTrade = async function(tradeObj) {
                console.log("acceptTrade function called");
                
                const action = await chooseAction();
                console.log("AI action for trade decision:", action);
                
                if (action === "do_nothing") {
                    return false;
                }
                
                // Calculate trade value
                var tradeValue = calculateTradeValue(tradeObj);
                var money = tradeObj.getMoney();
                var initiator = tradeObj.getInitiator();
                var recipient = tradeObj.getRecipient();
                var property = new Array(40).fill(0);
                
                for (var i = 0; i < 40; i++) {
                    property[i] = tradeObj.getProperty(i);
                }
                
                // Advanced trade logic
                if (action === "buy" && tradeValue > 0) {
                    return true;
                } else if (action === "sell" && tradeValue < 0) {
                    return false;
                }
                
                // Counter-offer if beneficial
                if (tradeValue > -50 && tradeValue < 15 && initiator.money > 25) {
                    var proposedMoney = 15 - tradeValue + money;
                    return new Trade(initiator, recipient, proposedMoney, property, 
                                    tradeObj.getCommunityChestJailCard(), 
                                    tradeObj.getChanceJailCard());
                }
                
                return tradeValue > 15; // Accept only very good deals
            }

            // Helper to calculate trade value - keeping original
            function calculateTradeValue(tradeObj) {
                let value = 0;
                
                // Value from money
                value += tradeObj.getMoney();
                
                // Value from jail cards
                value += 10 * tradeObj.getCommunityChestJailCard();
                value += 10 * tradeObj.getChanceJailCard();
                
                // Value from properties
                for (var i = 0; i < 40; i++) {
                    if (square[i].price) {
                        value += tradeObj.getProperty(i) * square[i].price * (square[i].mortgage ? 0.5 : 1);
                    }
                }
                
                return value;
            }

            // Main function to handle actions - similar to Python's handle_action
            this.handleAction = async function() {
                console.log("handleAction called");
                
                // Iterate through all property groups
                for (let groupIdx = 1; groupIdx <= 8; groupIdx++) {
                    if (this.isGroupActionable(groupIdx)) {
                        const action = await chooseAction();
                        console.log(`Action for group ${groupIdx}: ${action}`);
                        
                        this.actionSuccessful = await this.executeAction(action, groupIdx);
                        
                        if (this.actionSuccessful) {
                            break; // Exit after successful action
                        }
                    }
                }
                
                return this.actionSuccessful;
            };
            
            // Execute action based on action type - similar to Python's execute_action
            this.executeAction = async function(action, groupIdx) {
                console.log(`executeAction: ${action} for group ${groupIdx}`);
                
                if (action === "buy") {
                    return await this.buyInGroup(groupIdx);
                } else if (action === "sell") {
                    return await this.sellInGroup(groupIdx);
                } else if (action === "do_nothing") {
                    return true;
                }
                
                return false;
            };
            
            // Before turn actions - refactored to use handleAction
            this.beforeTurn = async function() {
                console.log("beforeTurn function called");
                return await this.handleAction();
            }
            
            // Buy properties in a group - similar to Python's buy_in_group
            this.buyInGroup = async function(groupIdx) {
                console.log(`buyInGroup for group ${groupIdx}`);
                const cellsInGroup = [];
                
                // Get all cells in this group
                for (let i = 0; i < 40; i++) {
                    if (square[i].groupNumber === parseInt(groupIdx)) {
                        cellsInGroup.push(i);
                    }
                }
                
                // Try to improve properties first (build houses/hotels)
                const propertyToImprove = this.getNextPropertyToImprove(cellsInGroup);
                if (propertyToImprove !== -1) {
                    return this.improveProperty(propertyToImprove);
                }
                
                // Try to unmortgage properties
                const propertyToUnmortgage = this.getNextPropertyToUnmortgage(cellsInGroup);
                if (propertyToUnmortgage !== -1) {
                    return this.unmortgageProperty(propertyToUnmortgage);
                }
                
                // Try to buy unowned property player is standing on
                if (this.canBuyCurrentProperty(cellsInGroup)) {
                    return this.buyCurrentProperty();
                }
                
                return false;
            };
            
            // Get next property to improve
            this.getNextPropertyToImprove = function(cellsInGroup) {
                const canBeImproved = [];
                
                for (const cellIdx of cellsInGroup) {
                    const cell = square[cellIdx];
                    
                    if (cell.owner !== p.index) continue;
                    if (cell.mortgage) continue;
                    if (cell.house === 5) continue; // Already has hotel
                    
                    // Check if it's part of a monopoly
                    if (!this.isMonopoly(cell.groupNumber)) continue;
                    
                    // Skip railroads and utilities
                    if (cell.groupNumber === 1 || cell.groupNumber === 2) continue;
                    
                    // Check if other properties in group have equal or more houses
                    let canImprove = true;
                    for (const otherCellIdx of cellsInGroup) {
                        if (otherCellIdx === cellIdx) continue;
                        
                        const otherCell = square[otherCellIdx];
                        if (otherCell.house < cell.house || otherCell.mortgage) {
                            canImprove = false;
                            break;
                        }
                    }
                    
                    if (canImprove && p.money > cell.houseprice + 100) {
                        canBeImproved.push(cellIdx);
                    }
                }
                
                // Sort by house price (cheapest first)
                canBeImproved.sort((a, b) => square[a].houseprice - square[b].houseprice);
                
                return canBeImproved.length > 0 ? canBeImproved[0] : -1;
            };
            
            // Improve a property (build house/hotel)
            this.improveProperty = function(propertyIndex) {
                const cell = square[propertyIndex];
                
                if (cell.house < 4) {
                    // Build a house
                    buyHouse(propertyIndex);
                    console.log(`Bought house on ${cell.name}`);
                    return true;
                } else if (cell.house === 4) {
                    // Build a hotel
                    buyHouse(propertyIndex);
                    console.log(`Bought hotel on ${cell.name}`);
                    return true;
                }
                
                return false;
            };
            
            // Get next property to unmortgage
            this.getNextPropertyToUnmortgage = function(cellsInGroup) {
                for (const cellIdx of cellsInGroup) {
                    const cell = square[cellIdx];
                    
                    if (cell.owner !== p.index || !cell.mortgage) continue;
                    
                    const unmortgageCost = Math.round(cell.price * 1.1);
                    if (p.money > unmortgageCost + 50) {
                        return cellIdx;
                    }
                }
                
                return -1;
            };
            
            // Unmortgage a property
            this.unmortgageProperty = function(propertyIndex) {
                unmortgage(propertyIndex);
                console.log(`Unmortgaged ${square[propertyIndex].name}`);
                return true;
            };
            
            // Check if player can buy current property
            this.canBuyCurrentProperty = function(cellsInGroup) {
                if (!cellsInGroup.includes(p.position)) return false;
                
                const currentSquare = square[p.position];
                return !currentSquare.owner && p.money > currentSquare.price + 50;
            };
            
            // Buy the current property
            this.buyCurrentProperty = function() {
                // The actual purchase is handled by the game engine
                console.log(`Planning to buy ${square[p.position].name}`);
                return true;
            };
            
            // Sell properties in a group - similar to Python's sell_in_group
            this.sellInGroup = async function(groupIdx) {
                console.log(`sellInGroup for group ${groupIdx}`);
                const cellsInGroup = [];
                
                // Get all cells in this group
                for (let i = 0; i < 40; i++) {
                    if (square[i].groupNumber === parseInt(groupIdx)) {
                        cellsInGroup.push(i);
                    }
                }
                
                // Try to downgrade properties first (sell houses/hotels)
                const propertyToDowngrade = this.getNextPropertyToDowngrade(cellsInGroup);
                if (propertyToDowngrade !== -1) {
                    return this.downgradeProperty(propertyToDowngrade);
                }
                
                // Try to mortgage properties
                const propertyToMortgage = this.getNextPropertyToMortgage(cellsInGroup);
                if (propertyToMortgage !== -1) {
                    return this.mortgageProperty(propertyToMortgage);
                }
                
                return false;
            };
            
            // Get next property to downgrade
            this.getNextPropertyToDowngrade = function(cellsInGroup) {
                const canBeDowngraded = [];
                
                for (const cellIdx of cellsInGroup) {
                    const cell = square[cellIdx];
                    
                    if (cell.owner !== p.index) continue;
                    if (cell.house === 0) continue;
                    
                    // Check if other properties in group have equal or fewer houses
                    let canDowngrade = true;
                    for (const otherCellIdx of cellsInGroup) {
                        if (otherCellIdx === cellIdx) continue;
                        
                        const otherCell = square[otherCellIdx];
                        if (otherCell.house > cell.house) {
                            canDowngrade = false;
                            break;
                        }
                    }
                    
                    if (canDowngrade) {
                        canBeDowngraded.push(cellIdx);
                    }
                }
                
                // Sort by development level (most developed first)
                canBeDowngraded.sort((a, b) => square[b].house - square[a].house);
                
                return canBeDowngraded.length > 0 ? canBeDowngraded[0] : -1;
            };
            
            // Downgrade a property (sell house/hotel)
            this.downgradeProperty = function(propertyIndex) {
                sellHouse(propertyIndex);
                console.log(`Sold house/hotel on ${square[propertyIndex].name}`);
                return true;
            };
            
            // Get next property to mortgage
            this.getNextPropertyToMortgage = function(cellsInGroup) {
                for (const cellIdx of cellsInGroup) {
                    const cell = square[cellIdx];
                    
                    if (cell.owner !== p.index || cell.mortgage || cell.house > 0) continue;
                    
                    // Don't mortgage if it's part of a monopoly
                    if (this.isMonopoly(cell.groupNumber)) continue;
                    
                    return cellIdx;
                }
                
                return -1;
            };
            
            // Mortgage a property
            this.mortgageProperty = function(propertyIndex) {
                mortgage(propertyIndex);
                console.log(`Mortgaged ${square[propertyIndex].name}`);
                return true;
            };

            // Actions when landing on a square - simplified to use handleAction
            this.onLand = async function() {
                console.log("onLand function called");
                return await this.handleAction();
            }

            // Decide whether to post bail
            this.postBail = async function() {
                console.log("postBail function called");
                
                const action = await chooseAction();
                console.log("AI action for jail decision:", action);
                
                // If we have action recommendation
                if (action === "buy") {
                    return true; // Pay to get out
                }
                
                // On third turn in jail, always try to get out
                if (p.jailroll === 2) {
                    return true;
                }
                
                // Use get out of jail card if we have one
                if (p.communityChestJailCard || p.chanceJailCard) {
                    return true;
                }
                
                // Otherwise stay in jail
                return false;
            }

            // Pay debt by mortgaging properties
            this.payDebt = async function() {
                console.log("payDebt function called");
                
                // Urgent situation - similar to Python, we sell first
                const sellOrder = this.calculateMortgagePriority();
                
                for (const index of sellOrder) {
                    const s = square[index];
                    
                    if (s.owner === p.index && !s.mortgage && s.house === 0) {
                        mortgage(index);
                        console.log(`Mortgaged ${s.name} to pay debt`);
                        
                        if (p.money >= 0) {
                            return;
                        }
                    }
                }
                
                // If still in debt and have houses/hotels, sell them
                for (var i = 0; i < 40; i++) {
                    const s = square[i];
                    
                    if (s.owner === p.index && s.house > 0) {
                        sellHouse(i);
                        console.log(`Sold house on ${s.name} to pay debt`);
                        
                        if (p.money >= 0) {
                            return;
                        }
                    }
                }
            }
            
            // Calculate priority order for mortgaging properties
            this.calculateMortgagePriority = function() {
                const properties = [];
                
                // Collect all properties owned by player
                for (let i = 0; i < 40; i++) {
                    if (square[i].owner === p.index && !square[i].mortgage && square[i].house === 0) {
                        properties.push({
                            index: i,
                            value: square[i].price,
                            isPartOfGroup: this.isPartOfMonopoly(square[i].groupNumber)
                        });
                    }
                }
                
                // Sort by priority (non-monopoly properties first, then by lowest value)
                properties.sort((a, b) => {
                    if (a.isPartOfGroup !== b.isPartOfGroup) {
                        return a.isPartOfGroup ? 1 : -1;
                    }
                    return a.value - b.value;
                });
                
                return properties.map(prop => prop.index);
            }

            // Bidding strategy for auctions
            this.bid = async function(property, currentBid) {
                console.log("bid function called for property:", property, "currentBid:", currentBid);
                
                const action = await chooseAction();
                console.log("AI action for auction:", action);
                
                const propertyObj = square[property];
                
                if (action === "buy") {
                    // More aggressive bidding
                    const maxBid = propertyObj.price * 1.2;
                    const bid = currentBid + Math.round(Math.random() * 30 + 10);
                    
                    if (p.money < bid + 100 || bid > maxBid) {
                        return -1; // Exit auction
                    } else {
                        return bid;
                    }
                } else if (action === "sell") {
                    // Exit auction
                    return -1;
                } else {
                    // Default bidding behavior
                    const maxBid = propertyObj.price * 0.8;
                    const bid = currentBid + Math.round(Math.random() * 20 + 5);
                    
                    if (p.money < bid + 200 || bid > maxBid) {
                        return -1; // Exit auction
                    } else {
                        return bid;
                    }
                }
            }

            // Resolve the constructor
            resolve(this);
        });
    }
    
    // Static count property for naming agents
    static count = 0;
}