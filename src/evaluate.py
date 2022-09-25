from sklearn.metrics import f1_score

def evaluation_f1(true_data, pred_data):
    true_data_list = true_data
    pred_data_list = pred_data

    ce_eval = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0
    }

    pipeline_eval = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0
    }

    for i in range(len(true_data_list)):
        # TP, FN checking
        is_ce_found = False
        is_pipeline_found = False
        for y_ano in true_data_list[i]['annotation']:
            y_category = y_ano[0]
            y_polarity = y_ano[2]

            for p_ano in pred_data_list[i]['annotation']:
                p_category = p_ano[0]
                p_polarity = p_ano[1]

                if y_category == p_category:
                    is_ce_found = True
                    if y_polarity == p_polarity:
                        is_pipeline_found = True
                    break

            if is_ce_found is True:
                ce_eval['TP'] += 1
            else:
                ce_eval['FN'] += 1

            if is_pipeline_found is True:
                pipeline_eval['TP'] += 1
            else:
                pipeline_eval['FN'] += 1

            is_ce_found = False
            is_pipeline_found = False

        # FP checking
        for p_ano in pred_data_list[i]['annotation']:
            p_category = p_ano[0]
            p_polarity = p_ano[1]

            for y_ano in true_data_list[i]['annotation']:
                y_category = y_ano[0]
                y_polarity = y_ano[2]

                if y_category == p_category:
                    is_ce_found = True
                    if y_polarity == p_polarity:
                        is_pipeline_found = True
                    break

            if is_ce_found is False:
                ce_eval['FP'] += 1

            if is_pipeline_found is False:
                pipeline_eval['FP'] += 1

    try:
        ce_precision = ce_eval['TP']/(ce_eval['TP']+ce_eval['FP'])
        ce_recall = ce_eval['TP']/(ce_eval['TP']+ce_eval['FN'])

        ce_result = {
            'Precision': ce_precision,
            'Recall': ce_recall,
            'F1': 2*ce_recall*ce_precision/(ce_recall+ce_precision)
        }
    except ZeroDivisionError:
        print("Error:")
        print("ce_eval_dict")
        print(ce_eval)
        exit(1)

    try:
        pipeline_precision = pipeline_eval['TP']/(pipeline_eval['TP']+pipeline_eval['FP'])
        pipeline_recall = pipeline_eval['TP']/(pipeline_eval['TP']+pipeline_eval['FN'])

        pipeline_result = {
            'Precision': pipeline_precision,
            'Recall': pipeline_recall,
            'F1': 2*pipeline_recall*pipeline_precision/(pipeline_recall+pipeline_precision)
        }
    except ZeroDivisionError:
        print("Error:")
        print("pipeline_eval_dict")
        print(pipeline_eval)
        exit(1)

    return {
        'category extraction result': ce_result,
        'entire pipeline result': pipeline_result
    }

def evaluation(y_true, y_pred, label_len):
    count_list = [0]*label_len
    hit_list = [0]*label_len
    for i in range(len(y_true)):
        count_list[y_true[i]] += 1
        if y_true[i] == y_pred[i]:
            hit_list[y_true[i]] += 1
    acc_list = []

    for i in range(label_len):
        acc_list.append(hit_list[i]/count_list[i])

    print(count_list)
    print(hit_list)
    print(acc_list)
    print('accuracy: ', (sum(hit_list) / sum(count_list)))
    print('macro_accuracy: ', sum(acc_list) / 3)
    # print(y_true)

    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))

    print('f1_score: ', f1_score(y_true, y_pred, average=None))
    print('f1_score_micro: ', f1_score(y_true, y_pred, average='micro'))
    print('f1_score_macro: ', f1_score(y_true, y_pred, average='macro'))