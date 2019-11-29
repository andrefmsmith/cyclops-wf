using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using System.Reactive;

[Combinator]
[Description("Given a sequence of checkpoints and the current location, emits a notification whenever the next checkpoint is reached.")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class CheckpointReached
{
    public IObservable<double> Process(IObservable<Tuple<double[], double>> source)
    {
        return Observable.Create<double>(observer =>
        {
            var nextCheckpoint = 0;
            var sourceObserver = Observer.Create<Tuple<double[], double>>(
                value =>
                {
                    var checkpoints = value.Item1;
                    var location = value.Item2;
                    if (nextCheckpoint >= checkpoints.Length) observer.OnCompleted();
                    else if (location >= checkpoints[nextCheckpoint])
                    {
                        observer.OnNext(checkpoints[nextCheckpoint++]);
                    }
                },
                observer.OnError,
                observer.OnCompleted);
            return source.SubscribeSafe(sourceObserver);
        });
    }
}
