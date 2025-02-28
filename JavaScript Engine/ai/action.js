export const Actions = ['buy', 'sell', 'do_nothing'];
export const Total_Actions = 3;
export class Action {
    constructor(action_type) {
        if (!Actions.includes(action_type)) {
            throw new Error(`Invalid action type: ${action_type}`);
        }
        this.action_type = action_type;
        this.action_index = Actions.indexOf(action_type);
    }

    toString() {
        return this.action_type;
    }
}
