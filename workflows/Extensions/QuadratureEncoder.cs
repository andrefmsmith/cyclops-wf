using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;

[Combinator]
[Description("Decodes the current state of a quadrature encoder from A/B channel inputs.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class QuadratureEncoder
{
    public QuadratureEncoder()
    {
        Step = 1;
    }

    [Description("The voltage threshold specifying high and low states in each encoder channel.")]
    public double VoltageThreshold { get; set; }

    [Description("The value of each encoder step increment.")]
    public int Step { get; set; } 

    public IObservable<int[]> Process(IObservable<Mat> source)
    {
        return Observable.Defer(() =>
        {
            var steps = 0;
            var prevA = false;
            var prevB = false;
            var first = true;
            return source.Select(value =>
            {
                var threshold = VoltageThreshold;
                var samples = new double[value.Rows * value.Cols];
                var result = new int[value.Cols];
                using(var sampleHeader = Mat.CreateMatHeader(samples, value.Rows, value.Cols, Depth.F64, 1))
                {
                    CV.Copy(value, sampleHeader);
                }

                for (int i = 0; i < value.Cols; i++)
                {
                    var a = samples[i] > threshold;
                    var b = samples[value.Cols + i] > threshold;
                    if (!first)
                    {
                        if (a & !prevA) // rising A
                        {
                            steps += b ? -Step : Step;
                        }
                        else if (b & !prevB) // rising B
                        {
                            steps += a ? Step : -Step;
                        }
                    }
                    else first = false;
                    prevA = a;
                    prevB = b;
                    result[i] = steps;
                }

                return result;
            });
        });
    }
}
