// ********************************************************
// *****    noonianJS little Neurons layers library   *****
// *****    By Patrick Raspino 07-18 Licence MIT      *****
// ********************************************************

neuralNetWork = function(lr = 0.05, scaleWeight = {min:-1,max:1}, fn = 'sigmoid') {
    this.scaleWeight = {min:scaleWeight.min, max:scaleWeight.max};
    this.fn = fn;
    this.learningRate = lr;
    this.layers = [];
    this.quadradicError = 0.0;
  }

  neuralNetWork.prototype.webWorker = function() {
    // WORKER FUNCTION
    function workerFunction() {
      var self = this;
        self.onmessage = function(e) {
          var neuralNetWork = new neuralNetWork();
            console.log('Received input: ', e.data.neuralNetWork); // message received from main thread
            self.postMessage({status:"calcul"});
            var cc = 0;
            for (var g=0; g<1000; g++) {
              self.postMessage({status:"val", val: neuralNetWork.layers[0].neurons[0].weight[0]});
            }
            neuralNetWork.propagate();
            self.postMessage({status:"fin"});
        }
    }
  //////////////// Worker ////////////////////////////
    this.dataObj = '(' + workerFunction + ')();';
    this.blob = new Blob([this.dataObj.replace('"use strict";', '')]);
    this.blobURL = (window.URL ? URL : webkitURL).createObjectURL(this.blob, {type: 'application/javascript; charset=utf-8'});
    this.worker = new Worker(this.blobURL); // spawn new worker
    this.worker.onmessage = function(e) {
      if (e.data.status == "calcul") console.log("CALCUL EN COURS");
      if (e.data.status == "fin") console.log("FIN DE CALCUL");
      if (e.data.status == "val") document.getElementById('error').innerHTML = e.data.val;
    };
  }

  neuralNetWork.prototype.addLayer = function(nbNeuron, nbInputNeuron) {
    var neurons = [];
    this.layers.push({neurons : []});
    var nb = nbInputNeuron;
    for (var i = 0; i < nbNeuron; i++) {
      neurons.push(new this.addNeuron(nb, this.scaleWeight));
    }
    this.layers[0].neurons = neurons;
  }

  neuralNetWork.prototype.addHiddenLayer = function(nbNeuron) {
    var nb = this.layers[this.layers.length-1].neurons.length;
    var neurons = [];
    this.layers.push({neurons : []});
    for (var i = 0; i < nbNeuron; i++) {
      neurons.push(new this.addNeuron(nb, this.scaleWeight));
    }
    this.layers[this.layers.length-1].neurons = neurons;
  }

  // INPUT NUMBERS SAME LIKE NB INPUT PER NEURON
  // INPUT [x,y,z, ...]
  neuralNetWork.prototype.propagate =  function(inputs, show = false) {
    for (var n = 0; n < this.layers[0].neurons.length; n++) {
      for (var i = 0; i < inputs.length; i++) {
        this.layers[0].neurons[n].input[i] = inputs[i];
      }
      this.layers[0].neurons[n].activation();
    }

    for (var l = 1;l < this.layers.length; l++) {
      for (var i = 0; i < this.layers[l].neurons.length; i++) {
        for (var j = 0; j < this.layers[l-1].neurons.length; j++) {
          this.layers[l].neurons[i].input[j] = this.layers[l-1].neurons[j].output;
        }
        this.layers[l].neurons[i].activation();
      }
    }

    //this.layers[l].neurons[n].output between 0-1 : ex => 0.85
  }

  // inputs / targets [[1,2,3], [4,7,8], [...]]
  neuralNetWork.prototype.train = function(inputs, targets, iteration = 20) {
    var quadradicError = 0.0;
    var error, neuron, output;
    if (iteration == 20) {this.epoch = 0;}

    // var train = ()=> {
    // for (var epoch = 0; epoch < iteration; epoch++) {
      for (var l = this.layers.length-1; l >= 0; l--) {
        if(!this.layers[l+1]) {
          for(var j = 0; j < this.layers[l].neurons.length; j++) {
							var neuron = this.layers[l].neurons[j];
							var output = neuron.output;
              // output * (1 - output) = sigmoid derivation || (targets[j] - output) = derivation output
							neuron.gradient = output * (1 - output) * (targets[j] - output);
              quadradicError += Math.pow((targets[j] - output), 2);
					}
          quadradicError = (quadradicError / this.layers[l].neurons.length) * 100;
        }
        else {
          for(j = 0; j < this.layers[l].neurons.length; j++) {
							neuron = this.layers[l].neurons[j];
							output = neuron.output;
							error = 0.0;
							for(k = 0; k < this.layers[l+1].neurons.length; k++) {
								error += this.layers[l+1].neurons[k].weight[j] * this.layers[l+1].neurons[k].gradient;
							}
							neuron.gradient = output * (1 - output) * error;
					}
        }
      }
      // BACKPROPAGATION
      for(l = 0; l < this.layers.length; l++) {
					for(j = 0; j < this.layers[l].neurons.length; j++) {
						neuron = this.layers[l].neurons[j];
						neuron.bias += this.learningRate * neuron.gradient;
						for(k = 0; k < neuron.weight.length; k++) {
							neuron.delta[k] = this.learningRate * neuron.gradient * (this.layers[l-1] ? this.layers[l-1].neurons[k].output : inputs[k]);
              neuron.weight[k] = parseFloat(neuron.weight[k] + neuron.delta[k]);
							neuron.weight[k] += neuron.momentum * neuron.previousDelta[k];
						}
						neuron.previousDelta = neuron.delta.slice();
					}
        }
        this.propagate(inputs);
    // }
    if (iteration > 0) {
      this.quadradicError = quadradicError.toFixed(2);
      document.getElementById('error').innerHTML = this.quadradicError+" %";
      console.log(this.quadradicError+" %");
      iteration--;
      this.train(inputs, targets, iteration);
    }
  }


  neuralNetWork.prototype.addNeuron = function(nbInput, scaleWeight, bias = 1) {
    this.bias = bias;
    this.output = 0;
    this.agregation = 0;
    this.gradient = 0;
    this.delta = new Array(nbInput);
    this.previousDelta = new Array(nbInput);
    this.input = new Array(nbInput);
    this.weight = new Array(nbInput);
    this.momentum = 0.5;
    this.drop = false;
    for (var i = 0; i < nbInput; i++) {
      this.input[i] = 0;
      this.weight[i] = rnd(scaleWeight.min, scaleWeight.max);
      this.delta[i] = 0;
      this.previousDelta[i] = 0;
    }

    this.activation = function() {
      var z = 0.0;
      for (var i in this.input) {
        z+= (this.input[i] * this.weight[i]);
      }
      z+= this.bias;
      this.agregation = z;
      this.output = sigmoid(z);
      return this.output;
    }
  }

  sigmoid = (value) => {
    return 1 / (1 + Math.exp(-value));
  }

  rnd =  (min, max, digit = 3) => {
    return (Math.random() * (max - min) + min).toFixed(digit);
  }
