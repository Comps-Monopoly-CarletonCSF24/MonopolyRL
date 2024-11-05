function setup_training(AI_agent){
	pcount = Total_Players_Training 

	var playerArray = new Array(pcount);
	var p;

	playerArray.randomize();

	for (var i = 1; i <= pcount - 1; i++) {
		p = player[playerArray[i - 1]];
		p.AI = new AITest(p);
	}
	//set the last agent to be different
	player_agent = player[playerArray[pcount]]
	player_agent.AI = new AI_agent(player_agent)

	play();
}

setup_training(AITest)