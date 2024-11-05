/**
 * Operations realated to community chest, chance, and the board
 */

import global_variables from "./global_variables.js";

function addamount(amount, cause) {
	var p = global_variables.player[turn];
	p.money += amount;
	addAlert(p.name + " received $" + amount + " from " + cause + ".");
}

function subtractamount(amount, cause) {
	var p = global_variables.player[turn];

	p.pay(amount, 0);

	addAlert(p.name + " lost $" + amount + " from " + cause + ".");
}

function luxurytax() {
	addAlert(global_variables.player[turn].name + " paid $100 for landing on Luxury Tax.");
	global_variables.player[turn].pay(100, 0);

	$("#landed").show().text("You landed on Luxury Tax. Pay $100.");
}

function citytax() {
	addAlert(global_variables.player[turn].name + " paid $200 for landing on City Tax.");
	global_variables.player[turn].pay(200, 0);

	$("#landed").show().text("You landed on City Tax. Pay $200.");
}

function payeachplayer(amount, cause) {
	var p = global_variables.player[turn];
	var total = 0;

	for (var i = 1; i <= global_variables.pcount; i++) {
		if (i != turn) {
			global_variables.player[i].money += amount;
			total += amount;
			creditor = p.money >= 0 ? i : creditor;

			p.pay(amount, creditor);
		}
	}

	addAlert(p.name + " lost $" + total + " from " + cause + ".");
}

function collectfromeachplayer(amount, cause) {
	var p = global_variables.player[turn];
	var total = 0;

	for (var i = 1; i <= global_variables.pcount; i++) {
		if (i != turn) {
			money = global_variables.player[i].money;
			if (money < amount) {
				p.money += money;
				total += money;
				global_variables.player[i].money = 0;
			} else {
				global_variables.player[i].pay(amount, turn);
				p.money += amount;
				total += amount;
			}
		}
	}

	addAlert(p.name + " received $" + total + " from " + cause + ".");
}

function advance(destination, pass) {
	var p = global_variables.player[turn];

	if (typeof pass === "number") {
		if (p.position < pass) {
			p.position = pass;
		} else {
			p.position = pass;
			p.money += 200;
			addAlert(p.name + " collected a $200 salary for passing GO.");
		}
	}
	if (p.position < destination) {
		p.position = destination;
	} else {
		p.position = destination;
		p.money += 200;
		addAlert(p.name + " collected a $200 salary for passing GO.");
	}

	land();
}

function advanceToNearestUtility() {
	var p = global_variables.player[turn];

	if (p.position < 12) {
		p.position = 12;
	} else if (p.position >= 12 && p.position < 28) {
		p.position = 28;
	} else if (p.position >= 28) {
		p.position = 12;
		p.money += 200;
		addAlert(p.name + " collected a $200 salary for passing GO.");
	}

	land(true);
}

function advanceToNearestRailroad() {
	var p = global_variables.player[turn];

	updatePosition();

	if (p.position < 15) {
		p.position = 15;
	} else if (p.position >= 15 && p.position < 25) {
		p.position = 25;
	} else if (p.position >= 35) {
		p.position = 5;
		p.money += 200;
		addAlert(p.name + " collected a $200 salary for passing GO.");
	}

	land(true);
}

function gotojail() {
	var p = global_variables.player[global_variables.turn];
	addAlert(p.name + " was sent directly to jail.");
	document.getElementById("landed").innerHTML = "You are in jail.";

	p.jail = true;
	global_variables.doublecount = 0;

	document.getElementById("nextbutton").value = "End turn";
	document.getElementById("nextbutton").title = "End turn and advance to the next player.";

	if (p.human) {
		document.getElementById("nextbutton").focus();
	}

	updatePosition();
	updateOwned();

	if (!p.human) {
		popup(p.AI.alertList, game.next);
		p.AI.alertList = "";
	}
}

function gobackthreespaces() {
	var p = global_variables.player[turn];

	p.position -= 3;

	land();
}

