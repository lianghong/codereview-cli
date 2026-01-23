// Example TypeScript file for testing
const MAGIC_NUMBER = 42;

interface Example {
    getValue(): number;
}

class ExampleImpl implements Example {
    constructor(private readonly value: number) {}

    getValue(): number {
        return this.value;
    }
}

export const createExample = (value: number): Example => {
    return new ExampleImpl(value);
};

const example = createExample(MAGIC_NUMBER);
console.log(example.getValue());
