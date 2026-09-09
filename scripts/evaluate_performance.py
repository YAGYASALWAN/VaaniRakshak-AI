"""Record detector performance from saved test results; never trains/tunes a model.

Run with --run-dir models/english_TIMESTAMP to use evaluation.json,
test_predictions.json and history.json. Aggregate-only reports are supported.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics as sk


def read_json(path):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def ratio(a, b):
    return a / b if b else None


def from_confusion(matrix):
    a = np.asarray(matrix)
    if a.shape != (2, 2) or not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all() or (a < 0).any() or not (a == np.floor(a)).all():
        raise ValueError('Confusion matrix must contain four nonnegative integer counts')
    tn, fp, fn, tp = map(int, a.ravel())
    n, negative, positive = tn+fp+fn+tp, tn+fp, fn+tp
    if not negative or not positive:
        raise ValueError('Both true classes are required')
    precision, recall, specificity = ratio(tp, tp+fp), tp/positive, tn/negative
    per_class = {}
    for name, correct, missed, wrong, support in [('genuine',tn,fp,fn,negative), ('synthetic',tp,fn,fp,positive)]:
        per_class[name] = {'precision':ratio(correct,correct+wrong), 'recall':correct/support,
                           'f1':ratio(2*correct,2*correct+missed+wrong), 'support':support}
    denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return {'n':n, 'confusion_matrix':[[tn,fp],[fn,tp]], 'accuracy':(tn+tp)/n,
            'precision':precision, 'recall':recall, 'f1':ratio(2*tp,2*tp+fp+fn),
            'specificity':specificity, 'false_positive_rate':fp/negative,
            'false_negative_rate':fn/positive, 'negative_predictive_value':ratio(tn,tn+fn),
            'balanced_accuracy':(recall+specificity)/2, 'mcc':ratio(tp*tn-fp*fn,denominator),
            'macro_f1':sum(v['f1'] for v in per_class.values())/2,
            'weighted_f1':sum(v['f1']*v['support'] for v in per_class.values())/n,
            'synthetic_prevalence':positive/n, 'per_class':per_class}


def validate_predictions(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Predictions must be a nonempty list')
    ids, labels, scores = set(), [], []
    for row in rows:
        key = row.get('id')
        y, p = row.get('label'), row.get('synthetic_score')
        if not isinstance(key, str) or not key or key in ids:
            raise ValueError('Prediction IDs must be unique nonempty strings')
        if type(y) is not int or y not in (0,1) or type(p) not in (float,int) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Expected label 0/1 and finite synthetic score in [0,1]')
        ids.add(key)
        labels.append(y)
        scores.append(p)
    if set(labels) != {0,1}:
        raise ValueError('Both true classes are required')
    return np.array(labels), np.array(scores, dtype=float)


def reliability(y, scores, bins=10):
    group = np.minimum((scores*bins).astype(int), bins-1)
    rows = []
    for i in range(bins):
        mask = group == i
        count = int(mask.sum())
        if count:
            rows.append({'bin':i,'count':count,'mean_score':float(scores[mask].mean()),
                         'synthetic_fraction':float(y[mask].mean())})
    return rows


def record(evaluation, predictions=None):
    saved = evaluation.get('test_metrics', evaluation)
    threshold = saved.get('threshold', .5)
    if type(threshold) not in (float,int) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Expected decision threshold between 0 and 1')
    curves = None
    if predictions is not None:
        y, scores = validate_predictions(predictions)
        cm = sk.confusion_matrix(y, scores >= threshold, labels=[0,1]).tolist()
        if 'confusion_matrix' in saved and cm != saved['confusion_matrix']:
            raise ValueError('Predictions disagree with saved confusion matrix; check run/threshold')
        result = from_confusion(cm)
        fpr,tpr,roc_thresholds = sk.roc_curve(y,scores,drop_intermediate=False)
        precision,recall,pr_thresholds = sk.precision_recall_curve(y,scores)
        calibration = reliability(y,scores)
        result.update({'roc_auc':float(sk.roc_auc_score(y,scores)),
                       'average_precision':float(sk.average_precision_score(y,scores)),
                       'brier_score':float(sk.brier_score_loss(y,scores)),
                       'log_loss':float(sk.log_loss(y,scores,labels=[0,1])),
                       'ece_10_bins':sum(r['count']*abs(r['mean_score']-r['synthetic_fraction']) for r in calibration)/len(y)})
        curves = {'y':y,'scores':scores,'fpr':fpr,'tpr':tpr,'roc_thresholds':roc_thresholds,
                  'pr_precision':precision,'pr_recall':recall,'pr_thresholds':pr_thresholds,
                  'calibration':calibration}
        evidence = 'Recomputed from saved per-recording predictions; no new model inference.'
    else:
        result = from_confusion(saved['confusion_matrix'])
        if 'roc_auc' in saved:
            value = saved['roc_auc']
            if type(value) not in (float,int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid reported ROC-AUC')
            result['roc_auc'] = value
        evidence = 'Counts-derived metrics; ROC-AUC, if present, is reported rather than recomputed.'
    if 'n' in saved and saved['n'] != result['n']:
        raise ValueError('Saved sample count differs from confusion matrix/predictions')
    # Aggregate values must agree with counts. Do not silently record stale results.
    for key in ('accuracy','precision','recall','f1','roc_auc'):
        if saved.get(key) is not None and result.get(key) is not None and not math.isclose(saved[key],result[key],rel_tol=1e-6,abs_tol=1e-8):
            raise ValueError(f'Saved {key} differs from recomputed result')
    return {'created_utc':datetime.now(timezone.utc).isoformat(), 'positive_class':'synthetic (1)',
            'negative_class':'genuine (0)', 'matrix_axes':'rows=true, columns=predicted; [[TN,FP],[FN,TP]]',
            'threshold':threshold,'evidence':evidence,'metrics':result,
            'source_context':{k:evaluation[k] for k in ('repository','revision','checkpoint','best_epoch','notice','provenance') if k in evaluation},
            'limitations':['These results apply to the recorded test set, not guaranteed real-world accuracy.',
                          'Synthetic scores are not established as calibrated probabilities.',
                          'The inconclusive UI band is not applied here: metrics use the binary threshold.',
                          'Undefined metrics are null; no test-set threshold or temperature fitting is performed.']}, curves


def table_csv(path, rows, fields):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def figures(output, report, curves, history):
    result, images = report['metrics'], []
    plt.rcParams.update({'font.size':10,'figure.dpi':140,'savefig.dpi':180})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output/name, bbox_inches='tight')
        plt.close(fig)
        images.append(name)
    cm = np.array(result['confusion_matrix'])
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, data, title, normalized in [(axes[0],cm,'Confusion matrix — counts',False),
            (axes[1],cm/cm.sum(axis=1,keepdims=True),'Confusion matrix — row percentages',True)]:
        im = ax.imshow(data,cmap='Blues',vmin=0,vmax=1 if normalized else None)
        fig.colorbar(im,ax=ax,fraction=.045)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Genuine','Synthetic'],yticklabels=['Genuine','Synthetic'],xlabel='Predicted class',ylabel='True class',title=title)
        for i in range(2):
            for j in range(2):
                text = f'{data[i,j]:.1%}' if normalized else f'{data[i,j]:,}'
                ax.text(j,i,text,ha='center',va='center',color='white' if data[i,j] > data.max()*.55 else 'black')
    save(fig,'confusion_heatmap.png')
    keys = ['accuracy','precision','recall','f1','specificity','balanced_accuracy']
    fig,ax=plt.subplots(figsize=(9,4))
    values=[result[k] if result[k] is not None else 0 for k in keys]
    bars=ax.bar([k.replace('_',' ').title() for k in keys],values,color='#236a89')
    ax.set(ylim=(0,1.15),ylabel='Metric value',title='Binary detector metrics — synthetic is the positive class')
    for bar,k,v in zip(bars,keys,values):
        ax.text(bar.get_x()+bar.get_width()/2,v+.02,'Undefined' if result[k] is None else f'{v:.2%}',ha='center',fontsize=9)
    save(fig,'metric_summary.png')
    if curves is not None:
        c=curves
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].plot(c['fpr'],c['tpr'],label=f"ROC-AUC {result['roc_auc']:.4f}")
        axes[0].plot([0,1],[0,1],'--',color='gray')
        axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC curve',xlim=(0,1),ylim=(0,1))
        axes[0].legend()
        axes[1].plot(c['pr_recall'],c['pr_precision'],label=f"Average precision {result['average_precision']:.4f}")
        axes[1].axhline(result['synthetic_prevalence'],ls='--',color='gray',label='Synthetic prevalence')
        axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall curve',xlim=(0,1),ylim=(0,1.02))
        axes[1].legend()
        save(fig,'roc_precision_recall.png')
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for label,name in [(0,'Genuine'),(1,'Synthetic')]:
            axes[0].hist(c['scores'][c['y']==label],bins=np.linspace(0,1,31),alpha=.55,density=True,label=name)
        axes[0].axvline(report['threshold'],color='black',ls='--',label='Decision threshold')
        axes[0].set(xlabel='Synthetic score',ylabel='Density',title='Score distributions')
        axes[0].legend()
        bins=c['calibration']
        axes[1].plot([0,1],[0,1],'--',color='gray',label='Ideal calibration')
        axes[1].plot([r['mean_score'] for r in bins],[r['synthetic_fraction'] for r in bins],'o-',label='Observed')
        axes[1].set(xlabel='Mean synthetic score',ylabel='Observed synthetic fraction',title='Reliability diagram — 10 bins',xlim=(0,1),ylim=(0,1))
        axes[1].legend()
        save(fig,'scores_calibration.png')
        table_csv(output/'calibration_bins.csv',bins,['bin','count','mean_score','synthetic_fraction'])
        table_csv(output/'roc_points.csv',[{'fpr':f,'tpr':t,'threshold':p} for f,t,p in zip(c['fpr'],c['tpr'],c['roc_thresholds'])],['fpr','tpr','threshold'])
        table_csv(output/'precision_recall_points.csv',[{'precision':p,'recall':r,'threshold':t} for p,r,t in zip(c['pr_precision'],c['pr_recall'],list(c['pr_thresholds'])+[None])],['precision','recall','threshold'])
    if history:
        epochs=[r['epoch'] for r in history]
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for key,name in [('train_loss','Training'),('dev_loss','Validation')]:
            if all(key in r for r in history):
                axes[0].plot(epochs,[r[key] for r in history],label=name)
        plotted=False
        for key,name in [('f1','Validation F1'),('roc_auc','Validation ROC-AUC')]:
            if all(key in r.get('dev_metrics',{}) for r in history):
                axes[1].plot(epochs,[r['dev_metrics'][key] for r in history],label=name)
                plotted=True
        axes[0].set(xlabel='Epoch',ylabel='Loss',title='Training history')
        if axes[0].lines: axes[0].legend()
        axes[1].set(xlabel='Epoch',ylabel='Metric',title='Validation history')
        if plotted: axes[1].legend()
        else: axes[1].text(.5,.5,'Validation metric history unavailable',ha='center',transform=axes[1].transAxes)
        save(fig,'training_history.png')
    return images


def generate(evaluation_path, output, predictions_path=None, history_path=None):
    evaluation=read_json(evaluation_path)
    predictions=read_json(predictions_path) if predictions_path else None
    history=read_json(history_path) if history_path else None
    report, curves=record(evaluation,predictions)
    report['inputs']={str(Path(p).name):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [evaluation_path,predictions_path,history_path] if p}
    report['missing_inputs']=[]
    if predictions is None:
        report['missing_inputs'].append('Per-recording predictions unavailable: ROC/PR curves, score distribution, Brier/log loss and calibration omitted.')
    if not history:
        report['missing_inputs'].append('Epoch history unavailable: training curves omitted.')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    images=figures(output,report,curves,history)
    report['charts']=images
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    flat=[{'metric':k,'value':v} for k,v in report['metrics'].items() if not isinstance(v,(list,dict))]
    table_csv(output/'metrics.csv',flat,['metric','value'])
    table_csv(output/'per_class.csv',[dict(class_name=k,**v) for k,v in report['metrics']['per_class'].items()],['class_name','precision','recall','f1','support'])
    rows=''.join(f'<tr><td>{html.escape(r["metric"])}</td><td>{"Undefined" if r["value"] is None else format(r["value"],".6g")}</td></tr>' for r in flat)
    notes=report['limitations']+report['missing_inputs']
    document='<!doctype html><html lang="en"><meta charset="utf-8"><title>VaaniRakshak performance record</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#172b3a}table{border-collapse:collapse}td,th{padding:8px 20px;border-bottom:1px solid #ccd6dd}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>VaaniRakshak performance record</h1>'
    document+=f'<p>{html.escape(report["evidence"])}</p><p>Positive class: synthetic. Threshold: {report["threshold"]}. Samples: {report["metrics"]["n"]:,}.</p>'
    document+='<pre>'+html.escape(json.dumps(report['source_context'],indent=2))+'</pre>'
    document+='<table><tr><th>Metric</th><th>Value</th></tr>'+rows+'</table>'
    document+=''.join(f'<figure><img src="{name}" alt="{name.replace("_"," ")}"></figure>' for name in images)
    document+='<h2>Scope and unavailable data</h2><ul>'+''.join('<li>'+html.escape(n)+'</li>' for n in notes)+'</ul></html>'
    (output/'report.html').write_text(document,encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir',type=Path)
    source.add_argument('--evaluation',type=Path)
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--history',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    evaluation=args.evaluation or args.run_dir/'evaluation.json'
    predictions=args.predictions
    history=args.history
    if args.run_dir:
        if predictions is None and (args.run_dir/'test_predictions.json').is_file(): predictions=args.run_dir/'test_predictions.json'
        if history is None and (args.run_dir/'history.json').is_file(): history=args.run_dir/'history.json'
    output=args.output or Path('data/reports/performance')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report=generate(evaluation,output,predictions,history)
    print('Report:',(output/'report.html').resolve())
    print('Metrics:',(output/'metrics.json').resolve())
    for note in report['missing_inputs']: print(note)


if __name__=='__main__':
    main()
