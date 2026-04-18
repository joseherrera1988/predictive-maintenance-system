# Architecture Diagrams

## 1. Package Overview — Files, Functions & Connections

All modules, their public functions, and every import dependency across the package.

```mermaid
flowchart LR

    subgraph CFG["configs/"]
        yaml["config.yaml
        ─────────────
        project · data
        features · ml_model
        dl_model · evaluation
        tracking"]
    end

    subgraph SRC["src/"]
        config["config.py
        ─────────────
        load_config()
        CONFIG"]

        pipeline["data_pipeline.py
        ─────────────
        load_raw_data()
        save_processed_data()
        run_pipeline()"]

        prep["preprocessing.py
        ─────────────
        drop_columns()
        encode_categoricals()
        scale_numericals()
        preprocess()"]

        train_ml["train_ml.py
        ─────────────
        train()
        → RandomForestClassifier"]

        train_dl["train_dl.py
        ─────────────
        LSTMClassifier
          __init__()
          forward()
        train()"]

        evaluate["evaluate.py
        ─────────────
        evaluate_model()"]

        tracker["tracker.py
        ─────────────
        log_experiment()"]

        model_utils["model_utils.py
        ─────────────
        save_model()
        load_model()
        _update_metadata()"]
    end

    subgraph ENTRY["Entry Points"]
        main["main.py
        ─────────────
        main()
        argparse:
          --data
          --model"]

        dash["dashboards/streamlit_app.py
        ─────────────
        Experiment Logs viewer
        Saved Models table
        Run Prediction uploader"]
    end

    subgraph TESTS["tests/"]
        t_prep["test_preprocessing.py
        ─────────────
        test_drop_columns_removes_nothing_by_default()
        test_preprocess_removes_nulls()
        test_preprocess_returns_dataframe()"]

        t_model["test_model.py
        ─────────────
        test_random_forest_trains_and_evaluates()
        test_evaluate_returns_all_keys()"]

        t_integ["test_integration.py
        ─────────────
        test_full_pipeline_end_to_end()"]

        t_perf["test_performance.py
        ─────────────
        test_model_f1_above_threshold()
        test_inference_speed()"]
    end

    subgraph FILES["Persistent Files"]
        raw["data/raw/*.csv"]
        processed["data/processed/*.csv"]
        saved["models/saved_models/*.pkl"]
        meta["models/metadata.json"]
        logs["experiments/logs.csv"]
    end

    %% config.yaml → config.py
    yaml --> config

    %% config.py used by most src modules
    config --> pipeline
    config --> prep
    config --> train_ml
    config --> train_dl
    config --> tracker
    config --> model_utils

    %% data_pipeline imports preprocessing
    pipeline --> prep

    %% data I/O
    raw --> pipeline
    pipeline --> processed

    %% train_ml imports
    train_ml --> evaluate
    train_ml --> model_utils
    train_ml --> tracker

    %% train_dl imports
    train_dl --> evaluate
    train_dl --> model_utils
    train_dl --> tracker

    %% model_utils writes files
    model_utils --> saved
    model_utils --> meta

    %% tracker writes file
    tracker --> logs

    %% main.py entry point
    main --> pipeline
    main --> train_ml
    main --> train_dl

    %% dashboard reads files
    dash --> logs
    dash --> meta

    %% tests
    t_prep --> prep
    t_model --> evaluate
    t_integ --> prep
    t_integ --> evaluate
    t_perf --> evaluate
```

---

## 2. Entry Points, Arguments & What's Behind Them

Every way to invoke the system, the arguments it accepts, and what executes.

```mermaid
flowchart LR

    subgraph CLI["CLI — python main.py"]
        arg_data["--data  required
        ─────────────────
        filename of CSV
        inside data/raw/"]

        arg_model["--model  default: ml
        ─────────────────
        ml   → RandomForest only
        dl   → LSTM only
        both → RandomForest + LSTM"]
    end

    subgraph PIPELINE_STEP["Data Pipeline"]
        rp["run_pipeline(filename)
        ① load_raw_data()
           reads data/raw/
        ② preprocess()
           drop · encode · scale · dropna
        ③ save_processed_data()
           writes data/processed/"]
    end

    subgraph ML_STEP["ML Training"]
        ml["train_ml.train(df)
        ① train_test_split
        ② RandomForestClassifier.fit()
        ③ evaluate_model()
        ④ save_model()
        ⑤ log_experiment()"]
    end

    subgraph DL_STEP["DL Training"]
        dl["train_dl.train(df)
        ① train_test_split
        ② DataLoader + TensorDataset
        ③ LSTMClassifier.forward()
        ④ Adam + BCELoss epochs
        ⑤ save_model()"]
    end

    subgraph OUTPUTS["Outputs written"]
        o1["models/saved_models/
        random_forest_TIMESTAMP.pkl
        lstm_TIMESTAMP.pkl"]
        o2["models/metadata.json
        name · path · saved_at"]
        o3["experiments/logs.csv
        run_id · timestamp · metrics"]
    end

    subgraph DASH_ENTRY["Dashboard — streamlit run dashboards/streamlit_app.py"]
        d1["Experiment Logs section
        reads experiments/logs.csv
        renders dataframe"]
        d2["Saved Models section
        reads models/metadata.json
        renders table"]
        d3["Run Prediction section
        file_uploader CSV input
        previews DataFrame
        stub: connect model manually"]
    end

    subgraph CFG_ENTRY["Config — configs/config.yaml"]
        c1["project.seed
        data.test_size / val_size
        features.target_column
        ml_model.params
        dl_model.params
        tracking.experiments_path
        tracking.models_path"]
    end

    arg_data --> rp
    arg_model --> |"ml or both"|ml
    arg_model --> |"dl or both"|dl

    rp --> ml
    rp --> dl

    ml --> o1
    ml --> o2
    ml --> o3

    dl --> o1
    dl --> o2

    CFG_ENTRY --> PIPELINE_STEP
    CFG_ENTRY --> ML_STEP
    CFG_ENTRY --> DL_STEP

    o3 --> d1
    o2 --> d2
```

---

## 3. Example User Flow — Common Interactions

End-to-end sequence for three typical scenarios: train ML, train DL, inspect results.

```mermaid
flowchart LR

    User(["User / Operator"])

    subgraph Flow1["Scenario A — Train ML model"]
        f1_cmd["python main.py
        --data sensors.csv
        --model ml"]

        f1_load["load_raw_data()
        reads data/raw/sensors.csv"]

        f1_prep["preprocess()
        drop_columns()
        encode_categoricals()
        scale_numericals()
        dropna()"]

        f1_save_proc["save_processed_data()
        writes data/processed/
        sensors.csv"]

        f1_split["train_test_split
        test_size=0.2  seed=42"]

        f1_fit["RandomForestClassifier.fit()
        n_estimators=100
        max_depth=10"]

        f1_eval["evaluate_model()
        accuracy · precision
        recall · f1 · roc_auc"]

        f1_save_m["save_model()
        random_forest_TIMESTAMP.pkl
        _update_metadata()"]

        f1_log["log_experiment()
        appends row to
        experiments/logs.csv"]
    end

    subgraph Flow2["Scenario B — Train DL model"]
        f2_cmd["python main.py
        --data sensors.csv
        --model dl"]

        f2_tensor["TensorDataset + DataLoader
        batch_size=32"]

        f2_lstm["LSTMClassifier
        hidden=64  layers=2
        dropout=0.2"]

        f2_train["Adam + BCELoss
        50 epochs"]

        f2_save["save_model()
        lstm_TIMESTAMP.pkl"]
    end

    subgraph Flow3["Scenario C — Inspect results in dashboard"]
        f3_cmd["streamlit run
        dashboards/streamlit_app.py"]

        f3_logs["Experiment Logs
        reads experiments/logs.csv
        shows run_id · metrics · params"]

        f3_models["Saved Models
        reads models/metadata.json
        shows name · path · saved_at"]

        f3_predict["Run Prediction
        upload CSV
        preview DataFrame"]
    end

    subgraph Flow4["Scenario D — Run both models"]
        f4_cmd["python main.py
        --data sensors.csv
        --model both"]
        f4_note["Runs Flow A then Flow B
        sequentially on same
        preprocessed DataFrame"]
    end

    User --> f1_cmd
    f1_cmd --> f1_load --> f1_prep --> f1_save_proc
    f1_save_proc --> f1_split --> f1_fit --> f1_eval
    f1_eval --> f1_save_m --> f1_log
    f1_log --> User

    User --> f2_cmd
    f2_cmd --> f1_load
    f1_prep --> f2_tensor --> f2_lstm --> f2_train --> f2_save
    f2_save --> User

    User --> f3_cmd
    f3_cmd --> f3_logs --> f3_models --> f3_predict
    f3_predict --> User

    User --> f4_cmd --> f4_note
    f4_note --> f1_fit
    f4_note --> f2_lstm
```
