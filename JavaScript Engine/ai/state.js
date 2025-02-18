export class State {
    constructor() {
        this.area = getArea();
        this.position = getPosition();
        this.finance = getFinance();
        this.state = getState(this.area, this.position, this.finance);
    }
}

const numPropertyPerGroup = {Brown: 2, Railroads: 4, Lightblue: 3, Pink: 3, Utilities: 4, Orange: 3, Red: 3, Yellow: 3, Green: 3, Indigo: 2 };

const groupIndicesPython = {Brown: 0, Railroads: 1, Lightblue: 2, Pink: 3, Utilities: 4, Orange: 5, Red: 6, Yellow: 7, Green: 8, Indigo: 9 };

//Remaps the group indices in the js engine to the indices in the python engine
const groupIndicesJStoGroupName = {0: "None", 1: "Railroads", 2: "Utilities", 3: "Brown", 4: "Lightblue", 5: "Pink", 6: "Orange", 7:"Red", 8:"Yellow", 9:"Green", 10:"Indigo"}

const Num_Groups = 10;
const LCM_Property_Per_Group = 12;
const Num_Total_Cells = 40;
const Total_Property_Points = 17;

function getArea() {
    const selfPropertyPoints = new Array(Num_Groups).fill(0);
    const othersPropertyPoints = new Array(Num_Groups).fill(0);
    
    for (let i = 0; i < Num_Total_Cells; i++) {
        let cell = square[i]
        if (cell.owner == 0) continue
        let groupIndex = cell.groupNumber;
        let groupName = groupIndicesJStoGroupName[groupIndex];
        let groupNumProperties = numPropertyPerGroup[groupName];
        let groupIndex_Python = groupIndicesPython[groupName];
        let cell_property_points =  LCM_Property_Per_Group / groupNumProperties;
        let group_property_points = 0;

        if (cell.hotel) {
            group_property_points = Total_Property_Points;
        }
        else if (cell.house) {
            group_property_points = LCM_Property_Per_Group + cell.house;
        }

        if (cell.owner == turn){
            if (group_property_points) {
                selfPropertyPoints[groupIndex_Python] = Math.max(selfPropertyPoints[groupIndex_Python], group_property_points)
            } else {
                selfPropertyPoints[groupIndex_Python] += cell_property_points
            }
        } else {
            if (group_property_points) {
                othersPropertyPoints[groupIndex_Python] = Math.max(othersPropertyPoints[groupIndex_Python], group_property_points)
            } else {
                othersPropertyPoints[groupIndex_Python] += cell_property_points
            }
        }

    }
    return [
        selfPropertyPoints.map(x => x / Total_Property_Points),
        othersPropertyPoints.map(x => x / Total_Property_Points)
    ];
}

function getPosition() {
    let positionInt = player[turn].position;
    return positionInt / (Num_Total_Cells - 1);
}

function getFinance() {
    var propertyOwnedCurrent = 0;
    var propertyOwnedTotal = 0;
    for(let i = 0; i < Num_Total_Cells; i++){
        let cell = square[i];
        if (cell.owner == turn) 
            propertyOwnedCurrent += cell.house + cell.hotel
        if (cell.owner != 0)
            propertyOwnedTotal += cell.house + cell.hotel
    }
    const propertyRatio = propertyOwnedTotal == 0 ? 0 : propertyOwnedCurrent / propertyOwnedTotal;

    const moneyNormalized = sigmoidMoney(player[turn].money);
    return [propertyRatio, moneyNormalized];
}

function sigmoidMoney(money) {
    return money / (1 + Math.abs(money));
}

export function getState(area, position, finance) {
    return [...area.flat(), position, ...finance];
}
