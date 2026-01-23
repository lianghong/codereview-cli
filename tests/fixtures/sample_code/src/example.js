// Example JavaScript file for testing
const MAGIC_NUMBER = 42;

class Example {
    constructor(value) {
        this.value = value;
    }

    getValue() {
        return this.value;
    }
}

const example = new Example(MAGIC_NUMBER);
console.log(example.getValue());
