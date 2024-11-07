class Observation{
    constructor(gameState, relativeAssets, relativeMoney, relativeArea){
        this.gameState = gameState;
        this.relativeAssets = relativeAssets;
        this.relativeMoney = relativeMoney;
        this.relativeArea = relativeArea;
    }

    equals(other){
        if(!(other instanceof Observation)) return false;

        return this.areaArraysEqual(this.gameState, other.gameState) &&
                this.relativeAssets == other.relativeAssets &&
                this.relat == other.relativeMoney &&
                this.relativeArea == other.realtiveArea;

    }

    areaArraysEqual(arr1,arr2){
        if(!Array.isArray(arr1) || !Array.isArray(arr2) || arr1.length !== arr2.length) return false;
        for (let i=0;i < arr1.length; i++){
            if (!Array.isArray(arr1[i]) || !Array.isArray(arr2[i]) || arr1[i].length !== arr2[i].length) return false;
            for (let j=0; j < arr1[i].length; j++){
                if (arr1[i][j] !== arr2[i][j]) return false;
            }
        }
    return true;
}
    static Builder = class{
        constructor(){
            this.state = null;
            this.assets = 0;
            this.money = 0;
            this.area = 0;
        }

        withAssets(assets){
            this.assets
        }
    }
}