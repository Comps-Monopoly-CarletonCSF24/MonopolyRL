/**
 * Global actions related to the game
 */

function roll() {
	var p = player[turn];

	$("#option").hide();
	$("#buy").show();
	$("#manage").hide();

	if (p.human) {
		document.getElementById("nextbutton").focus();
	}
	document.getElementById("nextbutton").value = "End turn";
	document.getElementById("nextbutton").title = "End turn and advance to the next player.";

	game.rollDice();
	var die1 = game.getDie(1);
	var die2 = game.getDie(2);

	doublecount++;

	if (die1 == die2) {
		addAlert(p.name + " rolled " + (die1 + die2) + " - doubles.");
	} else {
		addAlert(p.name + " rolled " + (die1 + die2) + ".");
	}

	if (die1 == die2 && !p.jail) {
		updateDice(die1, die2);

		if (doublecount < 3) {
			document.getElementById("nextbutton").value = "Roll again";
			document.getElementById("nextbutton").title = "You threw doubles. Roll again.";

		// If player rolls doubles three times in a row, send him to jail
		} else if (doublecount === 3) {
			p.jail = true;
			doublecount = 0;
			addAlert(p.name + " rolled doubles three times in a row.");
			updateMoney();


			if (p.human) {
				popup("You rolled doubles three times in a row. Go to jail.", gotojail);
			} else {
				gotojail();
			}

			return;
		}
	} else {
		document.getElementById("nextbutton").value = "End turn";
		document.getElementById("nextbutton").title = "End turn and advance to the next player.";
		doublecount = 0;
	}

	updatePosition();
	updateMoney();
	updateOwned();

	if (p.jail === true) {
		p.jailroll++;

		updateDice(die1, die2);
		if (die1 == die2) {
			document.getElementById("jail").style.border = "1px solid black";
			document.getElementById("cell11").style.border = "2px solid " + p.color;
			$("#landed").hide();

			p.jail = false;
			p.jailroll = 0;
			p.position = 10 + die1 + die2;
			doublecount = 0;

			addAlert(p.name + " rolled doubles to get out of jail.");

			land();
		} else {
			if (p.jailroll === 3) {

				if (p.human) {
					popup("<p>You must pay the $50 fine.</p>", function() {
						payfifty();
						player[turn].position=10 + die1 + die2;
						land();
					});
				} else {
					payfifty();
					p.position = 10 + die1 + die2;
					land();
				}
			} else {
				$("#landed").show();
				document.getElementById("landed").innerHTML = "You are in jail.";

				if (!p.human) {
					popup(p.AI.alertList, game.next);
					p.AI.alertList = "";
				}
			}
		}


	} else {
		updateDice(die1, die2);

		// Move player
		p.position += die1 + die2;

		// Collect $200 salary as you pass GO
		if (p.position >= 40) {
			p.position -= 40;
			p.money += 200;
			addAlert(p.name + " collected a $200 salary for passing GO.");
		}

		land();
	}
}

function updateDice() {
	var die0 = game.getDie(1);
	var die1 = game.getDie(2);

	$("#die0").show();
	$("#die1").show();

	if (document.images) {
		var element0 = document.getElementById("die0");
		var element1 = document.getElementById("die1");

		element0.classList.remove("die-no-img");
		element1.classList.remove("die-no-img");

		element0.title = "Die (" + die0 + " spots)";
		element1.title = "Die (" + die1 + " spots)";

		if (element0.firstChild) {
			element0 = element0.firstChild;
		} else {
			element0 = element0.appendChild(document.createElement("img"));
		}

		element0.src = "images/Die_" + die0 + ".png";
		element0.alt = die0;

		if (element1.firstChild) {
			element1 = element1.firstChild;
		} else {
			element1 = element1.appendChild(document.createElement("img"));
		}

		element1.src = "images/Die_" + die1 + ".png";
		element1.alt = die0;
	} else {
		document.getElementById("die0").textContent = die0;
		document.getElementById("die1").textContent = die1;

		document.getElementById("die0").title = "Die";
		document.getElementById("die1").title = "Die";
	}
}

function land(increasedRent) {
	increasedRent = !!increasedRent; // Cast increasedRent to a boolean value. It is used for the ADVANCE TO THE NEAREST RAILROAD/UTILITY Chance cards.

	var p = player[turn];
	var s = square[p.position];

	var die1 = game.getDie(1);
	var die2 = game.getDie(2);

	$("#landed").show();
	document.getElementById("landed").innerHTML = "You landed on " + s.name + ".";
	s.landcount++;
	addAlert(p.name + " landed on " + s.name + ".");

	// Allow player to buy the property on which he landed.
	if (s.price !== 0 && s.owner === 0) {

		if (!p.human) {

			if (p.AI.buyProperty(p.position)) {
				buy();
			}
		} else {
			document.getElementById("landed").innerHTML = "<div>You landed on <a href='javascript:void(0);' onmouseover='showdeed(" + p.position + ");' onmouseout='hidedeed();' class='statscellcolor'>" + s.name + "</a>.<input type='button' onclick='buy();' value='Buy ($" + s.price + ")' title='Buy " + s.name + " for " + s.pricetext + ".'/></div>";
		}


		game.addPropertyToAuctionQueue(p.position);
	}

	// Collect rent
	if (s.owner !== 0 && s.owner != turn && !s.mortgage) {
		var groupowned = true;
		var rent;

		// Railroads
		if (p.position == 5 || p.position == 15 || p.position == 25 || p.position == 35) {
			if (increasedRent) {
				rent = 25;
			} else {
				rent = 12.5;
			}

			if (s.owner == square[5].owner) {
				rent *= 2;
			}
			if (s.owner == square[15].owner) {
				rent *= 2;
			}
			if (s.owner == square[25].owner) {
				rent *= 2;
			}
			if (s.owner == square[35].owner) {
				rent *= 2;
			}

		} else if (p.position === 12) {
			if (increasedRent || square[28].owner == s.owner) {
				rent = (die1 + die2) * 10;
			} else {
				rent = (die1 + die2) * 4;
			}

		} else if (p.position === 28) {
			if (increasedRent || square[12].owner == s.owner) {
				rent = (die1 + die2) * 10;
			} else {
				rent = (die1 + die2) * 4;
			}

		} else {

			for (var i = 0; i < 40; i++) {
				sq = square[i];
				if (sq.groupNumber == s.groupNumber && sq.owner != s.owner) {
					groupowned = false;
				}
			}

			if (!groupowned) {
				rent = s.baserent;
			} else {
				if (s.house === 0) {
					rent = s.baserent * 2;
				} else {
					rent = s["rent" + s.house];
				}
			}
		}

		addAlert(p.name + " paid $" + rent + " rent to " + player[s.owner].name + ".");
		p.pay(rent, s.owner);
		player[s.owner].money += rent;

		document.getElementById("landed").innerHTML = "You landed on " + s.name + ". " + player[s.owner].name + " collected $" + rent + " rent.";
	} else if (s.owner > 0 && s.owner != turn && s.mortgage) {
		document.getElementById("landed").innerHTML = "You landed on " + s.name + ". Property is mortgaged; no rent was collected.";
	}

	// City Tax
	if (p.position === 4) {
		citytax();
	}

	// Go to jail. Go directly to Jail. Do not pass GO. Do not collect $200.
	if (p.position === 30) {
		updateMoney();
		updatePosition();

		if (p.human) {
			popup("<div>Go to jail. Go directly to Jail. Do not pass GO. Do not collect $200.</div>", gotojail);
		} else {
			gotojail();
		}

		return;
	}

	// Luxury Tax
	if (p.position === 38) {
		luxurytax();
	}

	updateMoney();
	updatePosition();
	updateOwned();

	if (!p.human) {
		popup(p.AI.alertList, chanceCommunityChest);
		p.AI.alertList = "";
	} else {
		chanceCommunityChest();
	}
}

function buy() {
	var p = player[turn];
	var property = square[p.position];
	var cost = property.price;

	if (p.money >= cost) {
		p.pay(cost, 0);

		property.owner = turn;
		updateMoney();
		addAlert(p.name + " bought " + property.name + " for " + property.pricetext + ".");

		updateOwned();

		$("#landed").hide();

	} else {
		popup("<p>" + p.name + ", you need $" + (property.price - p.money) + " more to buy " + property.name + ".</p>");
	}
}

async function chanceCommunityChest() {
	var p = player[turn];

	// Community Chest
	if (p.position === 2 || p.position === 17 || p.position === 33) {
		var communityChestIndex = communityChestCards.deck[communityChestCards.index];

		// Remove the get out of jail free card from the deck.
		if (communityChestIndex === 0) {
			communityChestCards.deck.splice(communityChestCards.index, 1);
		}

		popup("<img src='images/community_chest_icon.png' style='height: 50px; width: 53px; float: left; margin: 8px 8px 8px 0px;' /><div style='font-weight: bold; font-size: 16px; '>Community Chest:</div><div style='text-align: justify;'>" + communityChestCards[communityChestIndex].text + "</div>", function() {
			communityChestAction(communityChestIndex);
		});

		communityChestCards.index++;

		if (communityChestCards.index >= communityChestCards.deck.length) {
			communityChestCards.index = 0;
		}

	// Chance
	} else if (p.position === 7 || p.position === 22 || p.position === 36) {
		var chanceIndex = chanceCards.deck[chanceCards.index];

		// Remove the get out of jail free card from the deck.
		if (chanceIndex === 0) {
			chanceCards.deck.splice(chanceCards.index, 1);
		}

		popup("<img src='images/chance_icon.png' style='height: 50px; width: 26px; float: left; margin: 8px 8px 8px 0px;' /><div style='font-weight: bold; font-size: 16px; '>Chance:</div><div style='text-align: justify;'>" + chanceCards[chanceIndex].text + "</div>", function() {
			chanceAction(chanceIndex);
		});

		chanceCards.index++;

		if (chanceCards.index >= chanceCards.deck.length) {
			chanceCards.index = 0;
		}
	} else {
		if (!p.human) {
			p.AI.alertList = "";
			let onlandresults = await p.AI.onLand();
			if (!onlandresults) {
				game.next();
			}
		}
	}
}

function communityChestAction(communityChestIndex) {
	var p = player[turn]; // This is needed for reference in action() method.

	// $('#popupbackground').hide();
	// $('#popupwrap').hide();
	communityChestCards[communityChestIndex].action(p);

	updateMoney();

	if (communityChestIndex !== 15 && !p.human) {
		p.AI.alertList = "";
		game.next();
	}
}

function chanceAction(chanceIndex) {
	var p = player[turn]; // This is needed for reference in action() method.

	// $('#popupwrap').hide();
	chanceCards[chanceIndex].action(p);

	updateMoney();

	if (chanceIndex !== 15 && !p.human) {
		p.AI.alertList = "";
		game.next();
	}
}

async function play() {
	if (game.auction()) {
		return;
	}

	turn++;
	if (turn > pcount) {
		turn -= pcount;
		round++;
	}

	if (round > Max_Num_Rounds){
		var winner;
		winning_amount = -0x7f;
		for (var i = 0; i < pcount; i++){
			if(player[i].money > winning_amount){
				winner = player[i];
				winning_amount = player[i].money;
			} 
		}
		$("#control").hide();
		$("#board").hide();
		$("#refresh").show();
		popup("<p>Max number of rounds reached. Congratulations, " +  winner.name + ", you have won the game.</p><div>");
		return
	}

	var p = player[turn];
	game.resetDice();

	document.getElementById("pname").innerHTML = p.name;

	addAlert("Round " + round + ": It is " + p.name + "'s turn.");

	// Check for bankruptcy.
	p.pay(0, p.creditor);

	$("#landed, #option, #manage").hide();
	$("#board, #control, #moneybar, #viewstats, #buy").show();

	doublecount = 0;
	if (p.human) {
		document.getElementById("nextbutton").focus();
	}
	document.getElementById("nextbutton").value = "Roll Dice";
	document.getElementById("nextbutton").title = "Roll the dice and move your token accordingly.";

	$("#die0").hide();
	$("#die1").hide();

	if (p.jail) {
		$("#landed").show();
		document.getElementById("landed").innerHTML = "You are in jail.<input type='button' title='Pay $50 fine to get out of jail immediately.' value='Pay $50 fine' onclick='payfifty();' />";

		if (p.communityChestJailCard || p.chanceJailCard) {
			document.getElementById("landed").innerHTML += "<input type='button' id='gojfbutton' title='Use &quot;Get Out of Jail Free&quot; card.' onclick='useJailCard();' value='Use Card' />";
		}

		document.getElementById("nextbutton").title = "Roll the dice. If you throw doubles, you will get out of jail.";

		if (p.jailroll === 0)
			addAlert("This is " + p.name + "'s first turn in jail.");
		else if (p.jailroll === 1)
			addAlert("This is " + p.name + "'s second turn in jail.");
		else if (p.jailroll === 2) {
			document.getElementById("landed").innerHTML += "<div>NOTE: If you do not throw doubles after this roll, you <i>must</i> pay the $50 fine.</div>";
			addAlert("This is " + p.name + "'s third turn in jail.");
		}

		if (!p.human && p.AI.postBail()) {
			if (p.communityChestJailCard || p.chanceJailCard) {
				useJailCard();
			} else {
				payfifty();
			}
		}
	}

	updateMoney();
	updatePosition();
	updateOwned();

	$(".money-bar-arrow").hide();
	$("#p" + turn + "arrow").show();

	if (!p.human) {
		let beforeTurn = await p.AI.beforeTurn();
		if (!beforeTurn) {
			game.next();
		}
	}
}

class Game {
	constructor() {
		console.log("constructor called");
		return new Promise(async (resolve, reject) => {
			var die1;
			var die2;
			var areDiceRolled = false;
		
			this.auctionQueue = [];
			var highestbidder;
			var highestbid;
			var currentbidder = 1;
			var auctionproperty;
		
			this.rollDice = function() {
				die1 = Math.floor(Math.random() * 6) + 1;
				die2 = Math.floor(Math.random() * 6) + 1;
				areDiceRolled = true;
			};
		
			this.resetDice = function() {
				areDiceRolled = false;
			};
		
			this.next = async function() {
				if (!p.human && p.money < 0) {
					p.AI.payDebt();
		
					if (p.money < 0) {
						popup("<p>" + p.name + " is bankrupt. All of its assets will be turned over to " + player[p.creditor].name + ".</p>", game.bankruptcy);
					} else {
						console.log("rolling");
						roll();
					}
				} else if (areDiceRolled && doublecount === 0) {
					await play();
				} else {
					roll();
				}
			};
		
			this.getDie = function(die) {
				if (die === 1) {
					return die1;
				} else {
					return die2;
				}
		
			};
		
			// Auction functions:
		
			var finalizeAuction = async function() {
				var p = player[highestbidder];
				var sq = square[auctionproperty];
		
				if (highestbid > 0) {
					p.pay(highestbid, 0);
					sq.owner = highestbidder;
					addAlert(p.name + " bought " + sq.name + " for $" + highestbid + ".");
				}
		
				for (var i = 1; i <= pcount; i++) {
					player[i].bidding = true;
				}
		
				$("#popupbackground").hide();
				$("#popupwrap").hide();
		
				if (!game.auction()) {
					await play();
				}
			};
		
			this.addPropertyToAuctionQueue = function(propertyIndex) {
				this.auctionQueue.push(propertyIndex);
			};
		
			this.auction = function() {
				if (this.auctionQueue.length === 0) {
					return false;
				}
		
				let index = this.auctionQueue.shift();
		
				var s = square[index];
		
				if (s.price === 0 || s.owner !== 0) {
					return game.auction();
				}
		
				auctionproperty = index;
				highestbidder = 0;
				highestbid = 0;
				currentbidder = turn + 1;
		
				if (currentbidder > pcount) {
					currentbidder -= pcount;
				}
		
				popup("<div style='font-weight: bold; font-size: 16px; margin-bottom: 10px;'>Auction <span id='propertyname'></span></div><div>Highest Bid = $<span id='highestbid'></span> (<span id='highestbidder'></span>)</div><div><span id='currentbidder'></span>, it is your turn to bid.</div<div><input id='bid' title='Enter an amount to bid on " + s.name + ".' style='width: 291px;' /></div><div><input type='button' value='Bid' onclick='game.auctionBid();' title='Place your bid.' /><input type='button' value='Pass' title='Skip bidding this time.' onclick='game.auctionPass();' /><input type='button' value='Exit Auction' title='Stop bidding on " + s.name + " altogether.' onclick='if (confirm(\"Are you sure you want to stop bidding on this property altogether?\")) game.auctionExit();' /></div>", "blank");
		
				document.getElementById("propertyname").innerHTML = "<a href='javascript:void(0);' onmouseover='showdeed(" + auctionproperty + ");' onmouseout='hidedeed();' class='statscellcolor'>" + s.name + "</a>";
				document.getElementById("highestbid").innerHTML = "0";
				document.getElementById("highestbidder").innerHTML = "N/A";
				document.getElementById("currentbidder").innerHTML = player[currentbidder].name;
				document.getElementById("bid").onkeydown = function (e) {
					var key = 0;
					var isCtrl = false;
					var isShift = false;
		
					if (window.event) {
						key = window.event.keyCode;
						isCtrl = window.event.ctrlKey;
						isShift = window.event.shiftKey;
					} else if (e) {
						key = e.keyCode;
						isCtrl = e.ctrlKey;
						isShift = e.shiftKey;
					}
		
					if (isNaN(key)) {
						return true;
					}
		
					if (key === 13) {
						game.auctionBid();
						return false;
					}
		
					// Allow backspace, tab, delete, arrow keys, or if control was pressed, respectively.
					if (key === 8 || key === 9 || key === 46 || (key >= 35 && key <= 40) || isCtrl) {
						return true;
					}
		
					if (isShift) {
						return false;
					}
		
					// Only allow number keys.
					return (key >= 48 && key <= 57) || (key >= 96 && key <= 105);
				};
		
				document.getElementById("bid").onfocus = function () {
					this.style.color = "black";
					if (isNaN(this.value)) {
						this.value = "";
					}
				};
		
				updateMoney();
		
				if (!player[currentbidder].human) {
					currentbidder = turn; // auctionPass advances currentbidder.
					this.auctionPass();
				}
				return true;
			};
		
			this.auctionPass = function() {
				if (highestbidder === 0) {
					highestbidder = currentbidder;
				}
		
				while (true) {
					currentbidder++;
		
					if (currentbidder > pcount) {
						currentbidder -= pcount;
					}
		
					if (currentbidder == highestbidder) {
						finalizeAuction();
						return;
					} else if (player[currentbidder].bidding) {
						var p = player[currentbidder];
		
						if (!p.human) {
							var bid = p.AI.bid(auctionproperty, highestbid);
		
							if (bid === -1 || highestbid >= p.money) {
								p.bidding = false;
								addAlert(p.name + " exited the auction.");
								continue;
							} else if (bid === 0) {
								addAlert(p.name + " passed.");
								continue;
							} else if (bid > 0) {
								this.auctionBid(bid);
								addAlert(p.name + " bid $" + bid + ".");
								continue;
							}
							return;
						} else {
							break;
						}
					}
		
				}
		
				document.getElementById("currentbidder").innerHTML = player[currentbidder].name;
				document.getElementById("bid").value = "";
				document.getElementById("bid").style.color = "black";
			};
		
			this.auctionBid = function(bid) {
				bid = bid || parseInt(document.getElementById("bid").value, 10);
		
				if (bid === "" || bid === null) {
					document.getElementById("bid").value = "Please enter a bid.";
					document.getElementById("bid").style.color = "red";
				} else if (isNaN(bid)) {
					document.getElementById("bid").value = "Your bid must be a number.";
					document.getElementById("bid").style.color = "red";
				} else {
		
					if (bid > player[currentbidder].money) {
						document.getElementById("bid").value = "You don't have enough money to bid $" + bid + ".";
						document.getElementById("bid").style.color = "red";
					} else if (bid > highestbid) {
						highestbid = bid;
						document.getElementById("highestbid").innerHTML = parseInt(bid, 10);
						highestbidder = currentbidder;
						document.getElementById("highestbidder").innerHTML = player[highestbidder].name;
		
						document.getElementById("bid").focus();
		
						if (player[currentbidder].human) {
							this.auctionPass();
						}
					} else {
						document.getElementById("bid").value = "Your bid must be greater than highest bid. ($" + highestbid + ")";
						document.getElementById("bid").style.color = "red";
					}
				}
			};
		
			this.auctionExit = function() {
				player[currentbidder].bidding = false;
				this.auctionPass();
			};
		
			// Trade functions:
			var currentInitiator;
			var currentRecipient;
		
			// Define event handlers:
			var tradeMoneyOnKeyDown = function (e) {
				var key = 0;
				var isCtrl = false;
				var isShift = false;
		
				if (window.event) {
					key = window.event.keyCode;
					isCtrl = window.event.ctrlKey;
					isShift = window.event.shiftKey;
				} else if (e) {
					key = e.keyCode;
					isCtrl = e.ctrlKey;
					isShift = e.shiftKey;
				}
		
				if (isNaN(key)) {
					return true;
				}
		
				if (key === 13) {
					return false;
				}
		
				// Allow backspace, tab, delete, arrow keys, or if control was pressed, respectively.
				if (key === 8 || key === 9 || key === 46 || (key >= 35 && key <= 40) || isCtrl) {
					return true;
				}
		
				if (isShift) {
					return false;
				}
		
				// Only allow number keys.
				return (key >= 48 && key <= 57) || (key >= 96 && key <= 105);
			};
		
			var tradeMoneyOnFocus = function () {
				this.style.color = "black";
				if (isNaN(this.value) || this.value === "0") {
					this.value = "";
				}
			};
		
			var tradeMoneyOnChange = function(e) {
				$("#proposetradebutton").show();
				$("#canceltradebutton").show();
				$("#accepttradebutton").hide();
				$("#rejecttradebutton").hide();
		
				var amount = this.value;
		
				if (isNaN(amount)) {
					this.value = "This value must be a number.";
					this.style.color = "red";
					return false;
				}
		
				amount = Math.round(amount) || 0;
				this.value = amount;
		
				if (amount < 0) {
					this.value = "This value must be greater than 0.";
					this.style.color = "red";
					return false;
				}
		
				return true;
			};
		
			document.getElementById("trade-leftp-money").onkeydown = tradeMoneyOnKeyDown;
			document.getElementById("trade-rightp-money").onkeydown = tradeMoneyOnKeyDown;
			document.getElementById("trade-leftp-money").onfocus = tradeMoneyOnFocus;
			document.getElementById("trade-rightp-money").onfocus = tradeMoneyOnFocus;
			document.getElementById("trade-leftp-money").onchange = tradeMoneyOnChange;
			document.getElementById("trade-rightp-money").onchange = tradeMoneyOnChange;
		
			var resetTrade = function(initiator, recipient, allowRecipientToBeChanged) {
				var currentSquare;
				var currentTableRow;
				var currentTableCell;
				var currentTableCellCheckbox;
				var nameSelect;
				var currentOption;
				var allGroupUninproved;
				var currentName;
		
				var tableRowOnClick = function(e) {
					var checkboxElement = this.firstChild.firstChild;
		
					if (checkboxElement !== e.srcElement) {
						checkboxElement.checked = !checkboxElement.checked;
					}
		
					$("#proposetradebutton").show();
					$("#canceltradebutton").show();
					$("#accepttradebutton").hide();
					$("#rejecttradebutton").hide();
				};
		
				var initiatorProperty = document.getElementById("trade-leftp-property");
				var recipientProperty = document.getElementById("trade-rightp-property");
		
				currentInitiator = initiator;
				currentRecipient = recipient;
		
				// Empty elements.
				while (initiatorProperty.lastChild) {
					initiatorProperty.removeChild(initiatorProperty.lastChild);
				}
		
				while (recipientProperty.lastChild) {
					recipientProperty.removeChild(recipientProperty.lastChild);
				}
		
				var initiatorSideTable = document.createElement("table");
				var recipientSideTable = document.createElement("table");
		
		
				for (var i = 0; i < 40; i++) {
					currentSquare = square[i];
		
					// A property cannot be traded if any properties in its group have been improved.
					if (currentSquare.house > 0 || currentSquare.groupNumber === 0) {
						continue;
					}
		
					allGroupUninproved = true;
					var max = currentSquare.group.length;
					for (var j = 0; j < max; j++) {
		
						if (square[currentSquare.group[j]].house > 0) {
							allGroupUninproved = false;
							break;
						}
					}
		
					if (!allGroupUninproved) {
						continue;
					}
		
					// Offered properties.
					if (currentSquare.owner === initiator.index) {
						currentTableRow = initiatorSideTable.appendChild(document.createElement("tr"));
						currentTableRow.onclick = tableRowOnClick;
		
						currentTableCell = currentTableRow.appendChild(document.createElement("td"));
						currentTableCell.className = "propertycellcheckbox";
						currentTableCellCheckbox = currentTableCell.appendChild(document.createElement("input"));
						currentTableCellCheckbox.type = "checkbox";
						currentTableCellCheckbox.id = "tradeleftcheckbox" + i;
						currentTableCellCheckbox.title = "Check this box to include " + currentSquare.name + " in the trade.";
		
						currentTableCell = currentTableRow.appendChild(document.createElement("td"));
						currentTableCell.className = "propertycellcolor";
						currentTableCell.style.backgroundColor = currentSquare.color;
		
						if (currentSquare.groupNumber == 1 || currentSquare.groupNumber == 2) {
							currentTableCell.style.borderColor = "grey";
						} else {
							currentTableCell.style.borderColor = currentSquare.color;
						}
		
						currentTableCell.propertyIndex = i;
						currentTableCell.onmouseover = function() {showdeed(this.propertyIndex);};
						currentTableCell.onmouseout = hidedeed;
		
						currentTableCell = currentTableRow.appendChild(document.createElement("td"));
						currentTableCell.className = "propertycellname";
						if (currentSquare.mortgage) {
							currentTableCell.title = "Mortgaged";
							currentTableCell.style.color = "grey";
						}
						currentTableCell.textContent = currentSquare.name;
		
					// Requested properties.
					} else if (currentSquare.owner === recipient.index) {
						currentTableRow = recipientSideTable.appendChild(document.createElement("tr"));
						currentTableRow.onclick = tableRowOnClick;
		
						currentTableCell = currentTableRow.appendChild(document.createElement("td"));
						currentTableCell.className = "propertycellcheckbox";
						currentTableCellCheckbox = currentTableCell.appendChild(document.createElement("input"));
						currentTableCellCheckbox.type = "checkbox";
						currentTableCellCheckbox.id = "traderightcheckbox" + i;
						currentTableCellCheckbox.title = "Check this box to include " + currentSquare.name + " in the trade.";
		
						currentTableCell = currentTableRow.appendChild(document.createElement("td"));
						currentTableCell.className = "propertycellcolor";
						currentTableCell.style.backgroundColor = currentSquare.color;
		
						if (currentSquare.groupNumber == 1 || currentSquare.groupNumber == 2) {
							currentTableCell.style.borderColor = "grey";
						} else {
							currentTableCell.style.borderColor = currentSquare.color;
						}
		
						currentTableCell.propertyIndex = i;
						currentTableCell.onmouseover = function() {showdeed(this.propertyIndex);};
						currentTableCell.onmouseout = hidedeed;
		
						currentTableCell = currentTableRow.appendChild(document.createElement("td"));
						currentTableCell.className = "propertycellname";
						if (currentSquare.mortgage) {
							currentTableCell.title = "Mortgaged";
							currentTableCell.style.color = "grey";
						}
						currentTableCell.textContent = currentSquare.name;
					}
				}
		
				if (initiator.communityChestJailCard) {
					currentTableRow = initiatorSideTable.appendChild(document.createElement("tr"));
					currentTableRow.onclick = tableRowOnClick;
		
					currentTableCell = currentTableRow.appendChild(document.createElement("td"));
					currentTableCell.className = "propertycellcheckbox";
					currentTableCellCheckbox = currentTableCell.appendChild(document.createElement("input"));
					currentTableCellCheckbox.type = "checkbox";
					currentTableCellCheckbox.id = "tradeleftcheckbox40";
					currentTableCellCheckbox.title = "Check this box to include this Get Out of Jail Free Card in the trade.";
		
					currentTableCell = currentTableRow.appendChild(document.createElement("td"));
					currentTableCell.className = "propertycellcolor";
					currentTableCell.style.backgroundColor = "white";
					currentTableCell.style.borderColor = "grey";
		
					currentTableCell = currentTableRow.appendChild(document.createElement("td"));
					currentTableCell.className = "propertycellname";
		
					currentTableCell.textContent = "Get Out of Jail Free Card";
				} else if (recipient.communityChestJailCard) {
					currentTableRow = recipientSideTable.appendChild(document.createElement("tr"));
					currentTableRow.onclick = tableRowOnClick;
		
					currentTableCell = currentTableRow.appendChild(document.createElement("td"));
					currentTableCell.className = "propertycellcheckbox";
					currentTableCellCheckbox = currentTableCell.appendChild(document.createElement("input"));
					currentTableCellCheckbox.type = "checkbox";
					currentTableCellCheckbox.id = "traderightcheckbox40";
					currentTableCellCheckbox.title = "Check this box to include this Get Out of Jail Free Card in the trade.";
		
					currentTableCell = currentTableRow.appendChild(document.createElement("td"));
					currentTableCell.className = "propertycellcolor";
					currentTableCell.style.backgroundColor = "white";
					currentTableCell.style.borderColor = "grey";
		
					currentTableCell = currentTableRow.appendChild(document.createElement("td"));
					currentTableCell.className = "propertycellname";
		
					currentTableCell.textContent = "Get Out of Jail Free Card";
				}
		
				if (initiator.chanceJailCard) {
					currentTableRow = initiatorSideTable.appendChild(document.createElement("tr"));
					currentTableRow.onclick = tableRowOnClick;
		
					currentTableCell = currentTableRow.appendChild(document.createElement("td"));
					currentTableCell.className = "propertycellcheckbox";
					currentTableCellCheckbox = currentTableCell.appendChild(document.createElement("input"));
					currentTableCellCheckbox.type = "checkbox";
					currentTableCellCheckbox.id = "tradeleftcheckbox41";
					currentTableCellCheckbox.title = "Check this box to include this Get Out of Jail Free Card in the trade.";
		
					currentTableCell = currentTableRow.appendChild(document.createElement("td"));
					currentTableCell.className = "propertycellcolor";
					currentTableCell.style.backgroundColor = "white";
					currentTableCell.style.borderColor = "grey";
		
					currentTableCell = currentTableRow.appendChild(document.createElement("td"));
					currentTableCell.className = "propertycellname";
		
					currentTableCell.textContent = "Get Out of Jail Free Card";
				} else if (recipient.chanceJailCard) {
					currentTableRow = recipientSideTable.appendChild(document.createElement("tr"));
					currentTableRow.onclick = tableRowOnClick;
		
					currentTableCell = currentTableRow.appendChild(document.createElement("td"));
					currentTableCell.className = "propertycellcheckbox";
					currentTableCellCheckbox = currentTableCell.appendChild(document.createElement("input"));
					currentTableCellCheckbox.type = "checkbox";
					currentTableCellCheckbox.id = "traderightcheckbox41";
					currentTableCellCheckbox.title = "Check this box to include this Get Out of Jail Free Card in the trade.";
		
					currentTableCell = currentTableRow.appendChild(document.createElement("td"));
					currentTableCell.className = "propertycellcolor";
					currentTableCell.style.backgroundColor = "white";
					currentTableCell.style.borderColor = "grey";
		
					currentTableCell = currentTableRow.appendChild(document.createElement("td"));
					currentTableCell.className = "propertycellname";
		
					currentTableCell.textContent = "Get Out of Jail Free Card";
				}
		
				if (initiatorSideTable.lastChild) {
					initiatorProperty.appendChild(initiatorSideTable);
				} else {
					initiatorProperty.textContent = initiator.name + " has no properties to trade.";
				}
		
				if (recipientSideTable.lastChild) {
					recipientProperty.appendChild(recipientSideTable);
				} else {
					recipientProperty.textContent = recipient.name + " has no properties to trade.";
				}
		
				document.getElementById("trade-leftp-name").textContent = initiator.name;
		
				currentName = document.getElementById("trade-rightp-name");
		
				if (allowRecipientToBeChanged && pcount > 2) {
					// Empty element.
					while (currentName.lastChild) {
						currentName.removeChild(currentName.lastChild);
					}
		
					nameSelect = currentName.appendChild(document.createElement("select"));
					for (var i = 1; i <= pcount; i++) {
						if (i === initiator.index) {
							continue;
						}
		
						currentOption = nameSelect.appendChild(document.createElement("option"));
						currentOption.value = i + "";
						currentOption.style.color = player[i].color;
						currentOption.textContent = player[i].name;
		
						if (i === recipient.index) {
							currentOption.selected = "selected";
						}
					}
		
					nameSelect.onchange = function() {
						resetTrade(currentInitiator, player[parseInt(this.value, 10)], true);
					};
		
					nameSelect.title = "Select a player to trade with.";
				} else {
					currentName.textContent = recipient.name;
				}
		
				document.getElementById("trade-leftp-money").value = "0";
				document.getElementById("trade-rightp-money").value = "0";
		
			};
		
			var readTrade = function() {
				var initiator = currentInitiator;
				var recipient = currentRecipient;
				var property = new Array(40);
				var money;
				var communityChestJailCard;
				var chanceJailCard;
		
				for (var i = 0; i < 40; i++) {
		
					if (document.getElementById("tradeleftcheckbox" + i) && document.getElementById("tradeleftcheckbox" + i).checked) {
						property[i] = 1;
					} else if (document.getElementById("traderightcheckbox" + i) && document.getElementById("traderightcheckbox" + i).checked) {
						property[i] = -1;
					} else {
						property[i] = 0;
					}
				}
		
				if (document.getElementById("tradeleftcheckbox40") && document.getElementById("tradeleftcheckbox40").checked) {
					communityChestJailCard = 1;
				} else if (document.getElementById("traderightcheckbox40") && document.getElementById("traderightcheckbox40").checked) {
					communityChestJailCard = -1;
				} else {
					communityChestJailCard = 0;
				}
		
				if (document.getElementById("tradeleftcheckbox41") && document.getElementById("tradeleftcheckbox41").checked) {
					chanceJailCard = 1;
				} else if (document.getElementById("traderightcheckbox41") && document.getElementById("traderightcheckbox41").checked) {
					chanceJailCard = -1;
				} else {
					chanceJailCard = 0;
				}
		
				money = parseInt(document.getElementById("trade-leftp-money").value, 10) || 0;
				money -= parseInt(document.getElementById("trade-rightp-money").value, 10) || 0;
		
				var trade = new Trade(initiator, recipient, money, property, communityChestJailCard, chanceJailCard);
		
				return trade;
			};
		
			var writeTrade = function(tradeObj) {
				resetTrade(tradeObj.getInitiator(), tradeObj.getRecipient(), false);
		
				for (var i = 0; i < 40; i++) {
		
					if (document.getElementById("tradeleftcheckbox" + i)) {
						document.getElementById("tradeleftcheckbox" + i).checked = false;
						if (tradeObj.getProperty(i) === 1) {
							document.getElementById("tradeleftcheckbox" + i).checked = true;
						}
					}
		
					if (document.getElementById("traderightcheckbox" + i)) {
						document.getElementById("traderightcheckbox" + i).checked = false;
						if (tradeObj.getProperty(i) === -1) {
							document.getElementById("traderightcheckbox" + i).checked = true;
						}
					}
				}
		
				if (document.getElementById("tradeleftcheckbox40")) {
					if (tradeObj.getCommunityChestJailCard() === 1) {
						document.getElementById("tradeleftcheckbox40").checked = true;
					} else {
						document.getElementById("tradeleftcheckbox40").checked = false;
					}
				}
		
				if (document.getElementById("traderightcheckbox40")) {
					if (tradeObj.getCommunityChestJailCard() === -1) {
						document.getElementById("traderightcheckbox40").checked = true;
					} else {
						document.getElementById("traderightcheckbox40").checked = false;
					}
				}
		
				if (document.getElementById("tradeleftcheckbox41")) {
					if (tradeObj.getChanceJailCard() === 1) {
						document.getElementById("tradeleftcheckbox41").checked = true;
					} else {
						document.getElementById("tradeleftcheckbox41").checked = false;
					}
				}
		
				if (document.getElementById("traderightcheckbox41")) {
					if (tradeObj.getChanceJailCard() === -1) {
						document.getElementById("traderightcheckbox41").checked = true;
					} else {
						document.getElementById("traderightcheckbox41").checked = false;
					}
				}
		
				if (tradeObj.getMoney() > 0) {
					document.getElementById("trade-leftp-money").value = tradeObj.getMoney() + "";
				} else {
					document.getElementById("trade-rightp-money").value = (-tradeObj.getMoney()) + "";
				}
		
			};
		
			this.trade = function(tradeObj) {
				$("#board").hide();
				$("#control").hide();
				$("#trade").show();
				$("#proposetradebutton").show();
				$("#canceltradebutton").show();
				$("#accepttradebutton").hide();
				$("#rejecttradebutton").hide();
		
				if (tradeObj instanceof Trade) {
					writeTrade(tradeObj);
					this.proposeTrade();
				} else {
					var initiator = player[turn];
					var recipient = turn === 1 ? player[2] : player[1];
		
					currentInitiator = initiator;
					currentRecipient = recipient;
		
					resetTrade(initiator, recipient, true);
				}
			};
		
			this.cancelTrade = function() {
				$("#board").show();
				$("#control").show();
				$("#trade").hide();
		
		
				if (!player[turn].human) {
					player[turn].AI.alertList = "";
					game.next();
				}
		
			};
		
			this.acceptTrade = function(tradeObj) {
				if (isNaN(document.getElementById("trade-leftp-money").value)) {
					document.getElementById("trade-leftp-money").value = "This value must be a number.";
					document.getElementById("trade-leftp-money").style.color = "red";
					return false;
				}
		
				if (isNaN(document.getElementById("trade-rightp-money").value)) {
					document.getElementById("trade-rightp-money").value = "This value must be a number.";
					document.getElementById("trade-rightp-money").style.color = "red";
					return false;
				}
		
				var showAlerts = true;
				var money;
				var initiator;
				var recipient;
		
				if (tradeObj) {
					showAlerts = false;
				} else {
					tradeObj = readTrade();
				}
		
				money = tradeObj.getMoney();
				initiator = tradeObj.getInitiator();
				recipient = tradeObj.getRecipient();
		
		
				if (money > 0 && money > initiator.money) {
					document.getElementById("trade-leftp-money").value = initiator.name + " does not have $" + money + ".";
					document.getElementById("trade-leftp-money").style.color = "red";
					return false;
				} else if (money < 0 && -money > recipient.money) {
					document.getElementById("trade-rightp-money").value = recipient.name + " does not have $" + (-money) + ".";
					document.getElementById("trade-rightp-money").style.color = "red";
					return false;
				}
		
				var isAPropertySelected = 0;
		
				// Ensure that some properties are selected.
				for (var i = 0; i < 40; i++) {
					isAPropertySelected |= tradeObj.getProperty(i);
				}
		
				isAPropertySelected |= tradeObj.getCommunityChestJailCard();
				isAPropertySelected |= tradeObj.getChanceJailCard();
		
				if (isAPropertySelected === 0) {
					popup("<p>One or more properties must be selected in order to trade.</p>");
					return false;
				}
		
				if (showAlerts && !confirm(initiator.name + ", are you sure you want to make this exchange with " + recipient.name + "?")) {
					return false;
				}
		
				// Exchange properties
				for (var i = 0; i < 40; i++) {
		
					if (tradeObj.getProperty(i) === 1) {
						square[i].owner = recipient.index;
						addAlert(recipient.name + " received " + square[i].name + " from " + initiator.name + ".");
					} else if (tradeObj.getProperty(i) === -1) {
						square[i].owner = initiator.index;
						addAlert(initiator.name + " received " + square[i].name + " from " + recipient.name + ".");
					}
		
				}
		
				if (tradeObj.getCommunityChestJailCard() === 1) {
					initiator.communityChestJailCard = false;
					recipient.communityChestJailCard = true;
					addAlert(recipient.name + ' received a "Get Out of Jail Free" card from ' + initiator.name + ".");
				} else if (tradeObj.getCommunityChestJailCard() === -1) {
					initiator.communityChestJailCard = true;
					recipient.communityChestJailCard = false;
					addAlert(initiator.name + ' received a "Get Out of Jail Free" card from ' + recipient.name + ".");
				}
		
				if (tradeObj.getChanceJailCard() === 1) {
					initiator.chanceJailCard = false;
					recipient.chanceJailCard = true;
					addAlert(recipient.name + ' received a "Get Out of Jail Free" card from ' + initiator.name + ".");
				} else if (tradeObj.getChanceJailCard() === -1) {
					initiator.chanceJailCard = true;
					recipient.chanceJailCard = false;
					addAlert(initiator.name + ' received a "Get Out of Jail Free" card from ' + recipient.name + ".");
				}
		
				// Exchange money.
				if (money > 0) {
					initiator.pay(money, recipient.index);
					recipient.money += money;
		
					addAlert(recipient.name + " received $" + money + " from " + initiator.name + ".");
				} else if (money < 0) {
					money = -money;
		
					recipient.pay(money, initiator.index);
					initiator.money += money;
		
					addAlert(initiator.name + " received $" + money + " from " + recipient.name + ".");
				}
		
				updateOwned();
				updateMoney();
		
				$("#board").show();
				$("#control").show();
				$("#trade").hide();
		
				if (!player[turn].human) {
					player[turn].AI.alertList = "";
					game.next();
				}
			};
		
			this.proposeTrade = function() {
				if (isNaN(document.getElementById("trade-leftp-money").value)) {
					document.getElementById("trade-leftp-money").value = "This value must be a number.";
					document.getElementById("trade-leftp-money").style.color = "red";
					return false;
				}
		
				if (isNaN(document.getElementById("trade-rightp-money").value)) {
					document.getElementById("trade-rightp-money").value = "This value must be a number.";
					document.getElementById("trade-rightp-money").style.color = "red";
					return false;
				}
		
				var tradeObj = readTrade();
				var money = tradeObj.getMoney();
				var initiator = tradeObj.getInitiator();
				var recipient = tradeObj.getRecipient();
				var reversedTradeProperty = [];
		
				if (money > 0 && money > initiator.money) {
					document.getElementById("trade-leftp-money").value = initiator.name + " does not have $" + money + ".";
					document.getElementById("trade-leftp-money").style.color = "red";
					return false;
				} else if (money < 0 && -money > recipient.money) {
					document.getElementById("trade-rightp-money").value = recipient.name + " does not have $" + (-money) + ".";
					document.getElementById("trade-rightp-money").style.color = "red";
					return false;
				}
		
				var isAPropertySelected = 0;
		
				// Ensure that some properties are selected.
				for (var i = 0; i < 40; i++) {
					reversedTradeProperty[i] = -tradeObj.getProperty(i);
					isAPropertySelected |= tradeObj.getProperty(i);
				}
		
				isAPropertySelected |= tradeObj.getCommunityChestJailCard();
				isAPropertySelected |= tradeObj.getChanceJailCard();
		
				if (isAPropertySelected === 0) {
					popup("<p>One or more properties must be selected in order to trade.</p>");
		
					return false;
				}
		
				if (initiator.human && !confirm(initiator.name + ", are you sure you want to make this offer to " + recipient.name + "?")) {
					return false;
				}
		
				var reversedTrade = new Trade(recipient, initiator, -money, reversedTradeProperty, -tradeObj.getCommunityChestJailCard(), -tradeObj.getChanceJailCard());
		
				if (recipient.human) {
		
					writeTrade(reversedTrade);
		
					$("#proposetradebutton").hide();
					$("#canceltradebutton").hide();
					$("#accepttradebutton").show();
					$("#rejecttradebutton").show();
		
					addAlert(initiator.name + " initiated a trade with " + recipient.name + ".");
					popup("<p>" + initiator.name + " has proposed a trade with you, " + recipient.name + ". You may accept, reject, or modify the offer.</p>");
				} else {
					var tradeResponse = recipient.AI.acceptTrade(tradeObj);
		
					if (tradeResponse === true) {
						popup("<p>" + recipient.name + " has accepted your offer.</p>");
						this.acceptTrade(reversedTrade);
					} else if (tradeResponse === false) {
						popup("<p>" + recipient.name + " has declined your offer.</p>");
						return;
					} else if (tradeResponse instanceof Trade) {
						popup("<p>" + recipient.name + " has proposed a counteroffer.</p>");
						writeTrade(tradeResponse);
		
						$("#proposetradebutton, #canceltradebutton").hide();
						$("#accepttradebutton").show();
						$("#rejecttradebutton").show();
					}
				}
			};
		
			// Bankrupcy functions:
			this.eliminatePlayer = async function() {
				var p = player[turn];
		
				for (var i = p.index; i < pcount; i++) {
					player[i] = player[i + 1];
					player[i].index = i;
				}
		
				for (var i = 0; i < 40; i++) {
					if (square[i].owner >= p.index) {
						square[i].owner--;
					}
				}
		
				pcount--;
				turn--;
		
				if (pcount === 2) {
					document.getElementById("stats").style.width = "454px";
				} else if (pcount === 3) {
					document.getElementById("stats").style.width = "686px";
				}
		
				if (pcount === 1) {
					updateMoney();
					$("#control").hide();
					$("#board").hide();
					$("#refresh").show();
		
					// // Display land counts for survey purposes.
					// var text;
					// for (var i = 0; i < 40; i++) {
						// if (i === 0)
							// text = square[i].landcount;
						// else
							// text += " " + square[i].landcount;
					// }
					// document.getElementById("refresh").innerHTML += "<br><br><div><textarea type='text' style='width: 980px;' onclick='javascript:select();' />" + text + "</textarea></div>";
		
					popup("<p>Congratulations, " + player[1].name + ", you have won the game.</p><div>");
		
				} else {
					await play();
				}
			};
		
			this.bankruptcyUnmortgage = function() {
				var p = player[turn];
		
				if (p.creditor === 0) {
					game.eliminatePlayer();
					return;
				}
		
				var HTML = "<p>" + player[p.creditor].name + ", you may unmortgage any of the following properties, interest free, by clicking on them. Click OK when finished.</p><table>";
				var price;
		
				for (var i = 0; i < 40; i++) {
					sq = square[i];
					if (sq.owner == p.index && sq.mortgage) {
						price = Math.round(sq.price * 0.5);
		
						HTML += "<tr><td class='propertycellcolor' style='background: " + sq.color + ";";
		
						if (sq.groupNumber == 1 || sq.groupNumber == 2) {
							HTML += " border: 1px solid grey;";
						} else {
							HTML += " border: 1px solid " + sq.color + ";";
						}
		
						// Player already paid interest, so they can unmortgage for the mortgage price.
						HTML += "' onmouseover='showdeed(" + i + ");' onmouseout='hidedeed();'></td><td class='propertycellname'><a href='javascript:void(0);' title='Unmortgage " + sq.name + " for $" + price + ".' onclick='if (" + price + " <= player[" + p.creditor + "].money) {player[" + p.creditor + "].pay(" + price + ", 0); square[" + i + "].mortgage = false; addAlert(\"" + player[p.creditor].name + " unmortgaged " + sq.name + " for $" + price + ".\");} this.parentElement.parentElement.style.display = \"none\";'>Unmortgage " + sq.name + " ($" + price + ")</a></td></tr>";
		
						sq.owner = p.creditor;
		
					}
				}
		
				HTML += "</table>";
		
				popup(HTML, game.eliminatePlayer);
			};
		
			this.resign = function() {
				popup("<p>Are you sure you want to resign?</p>", game.bankruptcy, "Yes/No");
			};
		
			this.bankruptcy = function() {
				var p = player[turn];
				var pcredit = player[p.creditor];
				var bankruptcyUnmortgageFee = 0;
		
		
				if (p.money >= 0) {
					return;
				}
		
				addAlert(p.name + " is bankrupt.");
		
				if (p.creditor !== 0) {
					pcredit.money += p.money;
				}
		
				for (var i = 0; i < 40; i++) {
					sq = square[i];
					if (sq.owner == p.index) {
						// Mortgaged properties will be tranfered by bankruptcyUnmortgage();
						if (!sq.mortgage) {
							sq.owner = p.creditor;
						} else {
							bankruptcyUnmortgageFee += Math.round(sq.price * 0.1);
						}
		
						if (sq.house > 0) {
							if (p.creditor !== 0) {
								pcredit.money += sq.houseprice * 0.5 * sq.house;
							}
							sq.hotel = 0;
							sq.house = 0;
						}
		
						if (p.creditor === 0) {
							sq.mortgage = false;
							game.addPropertyToAuctionQueue(i);
							sq.owner = 0;
						}
					}
				}
		
				updateMoney();
		
				if (p.chanceJailCard) {
					p.chanceJailCard = false;
					pcredit.chanceJailCard = true;
				}
		
				if (p.communityChestJailCard) {
					p.communityChestJailCard = false;
					pcredit.communityChestJailCard = true;
				}
		
				if (pcount === 2 || bankruptcyUnmortgageFee === 0 || p.creditor === 0) {
					game.eliminatePlayer();
				} else {
					addAlert(pcredit.name + " paid $" + bankruptcyUnmortgageFee + " interest on the mortgaged properties received from " + p.name + ".");
					popup("<p>" + pcredit.name + ", you must pay $" + bankruptcyUnmortgageFee + " interest on the mortgaged properties you received from " + p.name + ".</p>", function() {player[pcredit.index].pay(bankruptcyUnmortgageFee, 0); game.bankruptcyUnmortgage();});
				}
			};
			resolve(this);
		});
	}
}

