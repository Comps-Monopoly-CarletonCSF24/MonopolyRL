// Whether the trade function is allowed
var ToggleTrade = false;

// The purpose of this AI is not to be a relistic opponant, but to give an example of a vaild AI player.
// This is an implementation of the fixed agent
// The p is player
square
function AITest(p) {
	this.alertList = "";
	// This variable is static, it is not related to each instance.
	this.constructor.count++;

	p.name = "AI Test " + this.constructor.count; // this gets your ai a proper 

	// Decide whether to buy a property the AI landed on.
	// Return: boolean (true to buy).
	// Arguments:
	// index: the property's index (0-39).

	// This fixed policy by the property if the AI has more than the price of the property + 50 bucks
	this.buyProperty = function(index) {
		console.log("buyProperty");
		var s = square[index]; // get the value of the square at the given index

		if (p.money > s.price + 50) {
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
		for (var i = 0; i < 40; i++) {
			property[i] = tradeObj.getProperty(i);
			tradeValue += tradeObj.getProperty(i) * square[i].price * (square[i].mortgage ? 0.5 : 1);
		}

		console.log(tradeValue);

		var proposedMoney = 25 - tradeValue + money; // trying to make 25 bucks off the trade. Will be useful in request

		// By any property that's offering you more than $25 backs??? Insane
		if (tradeValue > 25) {
			return true;
		// If they are requesting more than $50 and offering you more money than the 25 you wanted to save,
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
	// The purpose is to allow the AI to manage property and/or initiate trades. (why? Does every player get the same privileges? )
	// Return: boolean: Must return true if and only if the AI proposed a trade. (does the ai participate in trades that it did not propose?)
	this.onLand = function() {
		console.log("onLand");
		var proposedTrade;
		var property = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		var railroadIndexes = [5, 15, 25, 35]; // Why do we need railroads index? because it will come in handy when we have only one utility. Read below 
		var requestedRailroad;
		var offeredUtility;
		var s;

		// If AI owns exactly one utility, try to trade it for a railroad. (why is that ideal? why would you want to do that? Are railroads the most pricey things?)
		for (var i = 0; i < 4; i++) {
			s = square[railroadIndexes[i]];

			if (s.owner !== 0 && s.owner !== p.index) {
				requestedRailroad = s.index;
				break;
			}
		}


		// what is on square 12 or 18? (why should index own both and sell one he does not own?) => I guess this is where the trading happens no? 
		if (square[12].owner === p.index && square[28].owner !== p.index) {
			offeredUtility = 12;
		} else if (square[28].owner === p.index && square[12].owner !== p.index) {
			offeredUtility = 28;
		}

		// not offered the trade before? getDie???? This is where you make the trade. 
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
		if ((p.communityChestJailCard || p.chanceJailCard) && p.jailroll === 2) {
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

		bid = currentBid + Math.round(Math.random() * 20 + 10);	
		// only accepts bids that are about 0.5 more than the price of the property: this is an insane price to put up during a bid tbh
		if (p.money < bid + 50 || bid > square[property].price * 1.5) {
			return -1;
		} else {
			return bid;
		}

	}
	}
}
