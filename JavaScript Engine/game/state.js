class State {
    constructor(currentPlayer, players) {
        this.area = getArea(currentPlayer, players);
        this.position = getPosition(currentPlayer.position);
        this.finance = getFinance(currentPlayer, players);
        this.state = getState(this.area, this.position, this.finance);
    }
}

const numPropertyPerGroup = { Brown: 2, Railroads: 4, Lightblue: 3, Pink: 3, Utilities: 4, Orange: 3, Red: 3, Yellow: 3, Green: 3, Indigo: 2 };
const groupIndices = { Brown: 0, Railroads: 1, Lightblue: 2, Pink: 3, Utilities: 4, Orange: 5, Red: 6, Yellow: 7, Green: 8, Indigo: 9 };
const Num_Groups = 10;
const LCM_Property_Per_Group = 12;
const Num_Total_Cells = 40;
const Total_Property_Points = 17;

function getArea(currentPlayer, players) {
    const selfPropertyPoints = getPropertyPointsByGroup(currentPlayer);
    const othersPropertyPoints = new Array(Num_Groups).fill(0);
    
    for (const player of players) {
        if (!player.isBankrupt && player.name !== currentPlayer.name) {
            const points = getPropertyPointsByGroup(player);
            for (let i = 0; i < Num_Groups; i++) {
                othersPropertyPoints[i] += points[i];
            }
        }
    }
    return [
        selfPropertyPoints.map(x => x / Total_Property_Points),
        othersPropertyPoints.map(x => x / Total_Property_Points)
    ];
}

function getPropertyPointsByGroup(player) {
    const propertyByGroup = new Array(Num_Groups).fill(0);
    
    for (const property of player.owned) {
        const groupIndex = groupIndices[property.group];
        
        if (property.hasHotel > 0) {
            propertyByGroup[groupIndex] = Total_Property_Points;
        } else if (property.hasHouses > 0) {
            propertyByGroup[groupIndex] = Math.max(propertyByGroup[groupIndex], LCM_Property_Per_Group + property.hasHouses);
        } else if (propertyByGroup[groupIndex] < LCM_Property_Per_Group) {
            propertyByGroup[groupIndex] += LCM_Property_Per_Group / numPropertyPerGroup[property.group];
        }
    }
    return propertyByGroup;
}

function getPosition(positionInt) {
    return positionInt / (Num_Total_Cells - 1);
}

function getFinance(currentPlayer, players) {
    let propertyOwnedTotal = 0;
    
    for (const player of players) {
        if (!player.isBankrupt) {
            propertyOwnedTotal += getNumProperty(player);
        }
    }
    const propertyRatio = propertyOwnedTotal === 0 ? 0 : getNumProperty(currentPlayer) / propertyOwnedTotal;
    const moneyNormalized = sigmoidMoney(currentPlayer.money);
    return [propertyRatio, moneyNormalized];
}

function getNumProperty(player, houses = false) {
    let totalProperty = 0;
    
    for (const property of player.owned) {
        totalProperty += 1;
        if (houses) {
            totalProperty += property.hasHotel + property.hasHouses;
        }
    }
    return totalProperty;
}

function sigmoidMoney(money) {
    return money / (1 + Math.abs(money));
}

function getState(area, position, finance) {
    return [...area.flat(), position, ...finance];
}
