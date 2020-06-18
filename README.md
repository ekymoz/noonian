Example

t = new neuralNetWork(0.07);
t.addLayer(5,10); // 5 entries - 20 outputs
t.addHiddenLayer(20);
t.addHiddenLayer(10);

// 3 Layers
  - first : 5 neurons
  - second : 20 neurons
  - third : 10 neurons

t.train([x,x,x,x,x], [x,x,x,x,x], [x,x,x,x,x], .....); // for training network (pattern), Here 5 values per Epoch


t.propagate([pattern], true);

Values of outputs between 0-1
t.layers[here 0-2].neurons[i (if layers=2 => i 0-9)].output



Easy !
