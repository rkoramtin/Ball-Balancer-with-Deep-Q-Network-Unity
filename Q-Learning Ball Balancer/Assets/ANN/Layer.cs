using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layer {

	public int numNeurons;
	public List<Neuron> neurons = new List<Neuron>();

	public Layer(int nNeurons, int numNeuronInputs)
	{
		numNeurons = nNeurons;
		for(int i = 0; i < nNeurons; i++)
		{
			neurons.Add(new Neuron(numNeuronInputs));
		}
	}
}
