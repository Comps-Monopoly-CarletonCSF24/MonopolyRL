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
                //return await this.handleAction();
                //the beforeturn function always returns false
                return false
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
                await this.handleAction();
                return false;
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