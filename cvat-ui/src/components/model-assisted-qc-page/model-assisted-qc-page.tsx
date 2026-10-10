import './styles.scss';
import React, { useEffect, useState } from 'react';
import { useHistory, useParams } from 'react-router';
import Axios from 'axios';
import { LeftOutlined, ExperimentOutlined } from '@ant-design/icons';
import Alert from 'antd/lib/alert';
import Button from 'antd/lib/button';
import Card from 'antd/lib/card';
import Form from 'antd/lib/form';
import Input from 'antd/lib/input';
import InputNumber from 'antd/lib/input-number';
import { Col, Row } from 'antd/lib/grid';
import Select from 'antd/lib/select';
import Space from 'antd/lib/space';
import Table from 'antd/lib/table';
import Tag from 'antd/lib/tag';
import Typography from 'antd/lib/typography';
import notification from 'antd/lib/notification';

interface EvaluationRun {
    id: number;
    status: string;
    model_path: string;
    metrics: Record<string, number>;
    error: string;
}

interface Overview {
    dataset: null | {
        ground_truth_job_id: number;
        sampled_frames: number[];
        conditions: object;
        validation_status: 'not_validated' | 'passed' | 'warning' | 'blocked';
        validation_result: {
            summary?: { frames_checked: number; boxes_checked: number; blockers: number; warnings: number };
            issues?: GroundTruthIssue[];
        };
    };
    runs: EvaluationRun[];
}

interface GroundTruthIssue {
    frame: number;
    severity: 'blocker' | 'warning';
    rule: string;
    message: string;
    shape_id?: number;
}

interface ThresholdResult {
    confidence: number;
    iou_threshold: number;
    map50: number;
    map50_95: number;
    precision: number;
    recall: number;
    f1: number;
    conditions: Record<string, any>;
}

interface ThresholdAnalysis {
    id: number;
    status: string;
    model_path: string;
    objective: string;
    conditions: Record<string, any>;
    results: ThresholdResult[];
    best_result: Partial<ThresholdResult>;
    error: string;
}

interface ConditionSummary extends Omit<ThresholdResult, 'confidence' | 'iou_threshold' | 'conditions'> {
    conditions: Record<string, any>;
    runs: number;
}

interface QCAlert {
    id: number;
    evaluation_run_id: number;
    frame: number;
    image_name: string;
    type: string;
    prediction: { class_name: string; confidence: number };
    annotation: null | { class_name: string };
    iou: null | number;
    risk_score: number;
    reason_codes: string[];
    status: string;
}

interface QCBenchmark {
    id: number;
    evaluation_run_id: number;
    metrics: {
        baselines: Record<string, { qc_precision: number; qc_recall: number; precision_at_k: Record<string, number> }>;
    };
}

export default function ModelAssistedQCPage(): JSX.Element {
    const { tid } = useParams<{ tid: string }>();
    const history = useHistory();
    const [overview, setOverview] = useState<Overview>({ dataset: null, runs: [] });
    const [analyses, setAnalyses] = useState<ThresholdAnalysis[]>([]);
    const [conditionSummaries, setConditionSummaries] = useState<ConditionSummary[]>([]);
    const [alerts, setAlerts] = useState<QCAlert[]>([]);
    const [benchmarks, setBenchmarks] = useState<QCBenchmark[]>([]);
    const [loading, setLoading] = useState(false);
    const [samplingMethod, setSamplingMethod] = useState('random_uniform');
    const endpoint = `/api/model-assisted-qc/tasks/${tid}`;
    const load = (): Promise<void> => Axios.get(`${endpoint}/dataset`).then(({ data }) => setOverview(data));
    const loadAnalyses = (): Promise<void> => Axios.get(`${endpoint}/threshold-analyses`).then(
        ({ data }) => setAnalyses(data),
    );
    const loadConditionSummaries = (): Promise<void> => Axios.get('/api/model-assisted-qc/condition-summary').then(
        ({ data }) => setConditionSummaries(data),
    );
    const loadAlerts = (): Promise<void> => Axios.get(`${endpoint}/alerts`).then(({ data }) => setAlerts(data));
    const loadBenchmarks = (): Promise<void> => Axios.get(`${endpoint}/benchmarks`).then(
        ({ data }) => setBenchmarks(data),
    );

    useEffect(() => {
        Promise.all([load(), loadAnalyses(), loadConditionSummaries(), loadAlerts(), loadBenchmarks()])
            .catch(() => undefined);
    }, [tid]);

    const submit = async (url: string, values: Record<string, any>): Promise<void> => {
        setLoading(true);
        try {
            await Axios.post(url, values);
            notification.success({ message: 'Operation completed' });
            await Promise.all([load(), loadAnalyses(), loadConditionSummaries(), loadAlerts(), loadBenchmarks()]);
        } catch (error: any) {
            notification.error({
                message: 'Operation failed',
                description: error.response?.data?.error || error.message,
            });
            await Promise.all([load(), loadAnalyses(), loadConditionSummaries(), loadAlerts(), loadBenchmarks()]);
        } finally { setLoading(false); }
    };

    const createDataset = (values: Record<string, any>): void => {
        try {
            submit(`${endpoint}/dataset`, { ...values, conditions: JSON.parse(values.conditions || '{}') });
        } catch (error: any) {
            notification.error({
                message: 'Invalid environment metadata JSON',
                description: error.message,
            });
        }
    };

    const validationStatus = overview.dataset?.validation_status || 'not_validated';
    const validationSummary = overview.dataset?.validation_result?.summary;
    const validationIssues = overview.dataset?.validation_result?.issues || [];
    const canEvaluate = validationStatus === 'passed' || validationStatus === 'warning';
    const validationColor = validationStatus === 'passed' ?
        'green' : validationStatus === 'warning' ? 'orange' : 'red';
    const latestAnalysis = analyses[0];
    const latestCompletedRun = overview.runs.find((run) => run.status === 'completed');
    const latestBenchmark = benchmarks[0]?.metrics?.baselines?.risk_ranked;
    const parseThresholds = (value: string): number[] => value.split(',').map(Number).filter(Number.isFinite);
    const bestObjectiveValue = latestAnalysis?.best_result[
        latestAnalysis.objective as keyof ThresholdResult
    ] as number | undefined;
    const bestSummary = latestAnalysis ? [
        `Best: Confidence ${latestAnalysis.best_result.confidence}`,
        `IoU ${latestAnalysis.best_result.iou_threshold}`,
        `${latestAnalysis.objective} ${bestObjectiveValue?.toFixed(4)}`,
    ].join(', ') : '';

    return (
        <Row justify='center' className='cvat-model-assisted-qc-page'>
            <Col span={22} xl={18}>
                <Button
                    type='link'
                    icon={<LeftOutlined />}
                    onClick={() => history.push(`/tasks/${tid}`)}
                >
                    Back to task
                </Button>
                <Typography.Title level={2}><ExperimentOutlined /> Ground Truth & YOLO Setup</Typography.Title>
                <Row gutter={[16, 16]}>
                    <Col xs={24} lg={12}>
                        <Card title='Phase 1 — Dataset & Ground Truth'>
                            {overview.dataset && (
                                <Space direction='vertical' style={{ width: '100%', marginBottom: 16 }}>
                                    <Alert
                                        type='success'
                                        showIcon
                                        message={`Ground Truth dataset is ready — ${overview.dataset.sampled_frames.length} frames`}
                                    />
                                    <Space>
                                        <Button
                                            loading={loading}
                                            onClick={() => submit(`${endpoint}/validate-ground-truth`, {})}
                                        >
                                            Validate Ground Truth
                                        </Button>
                                        <Tag color={validationColor}>{validationStatus}</Tag>
                                    </Space>
                                    {validationSummary && (
                                        <Alert
                                            type={validationStatus === 'blocked' ?
                                                'error' : validationStatus === 'warning' ? 'warning' : 'success'}
                                            showIcon
                                            message={`${validationSummary.frames_checked} frames, ${validationSummary.boxes_checked} boxes, ${validationSummary.blockers} blockers, ${validationSummary.warnings} warnings`}
                                        />
                                    )}
                                </Space>
                            )}
                            <Form
                                layout='vertical'
                                onFinish={createDataset}
                                initialValues={{
                                    sampling_method: 'random_uniform',
                                    frame_count: 100,
                                    random_seed: 0,
                                    conditions: '{}',
                                }}
                            >
                                <Form.Item name='sampling_method' label='Sampling method'>
                                    <Select
                                        onChange={setSamplingMethod}
                                        options={[
                                            { value: 'random_uniform', label: 'Random uniform' },
                                            { value: 'manual', label: 'Manual frame list' },
                                        ]}
                                    />
                                </Form.Item>
                                {samplingMethod === 'random_uniform' ? (
                                    <>
                                        <Form.Item name='frame_count' label='Number of frames'>
                                            <InputNumber min={1} />
                                        </Form.Item>
                                        <Form.Item name='random_seed' label='Random seed'>
                                            <InputNumber min={0} />
                                        </Form.Item>
                                    </>
                                ) : (
                                    <Form.Item
                                        name='frames'
                                        label='Manual frame numbers (comma-separated)'
                                        extra='Example: 0, 10, 25. Frame numbers start at 0.'
                                        getValueFromEvent={(event) => event.target.value
                                            .split(',')
                                            .map((value: string) => value.trim())
                                            .filter((value: string) => /^\d+$/.test(value))
                                            .map(Number)}
                                    >
                                        <Input placeholder='0, 10, 25' />
                                    </Form.Item>
                                )}
                                <Form.Item name='conditions' label='Environment metadata (JSON)'>
                                    <Input.TextArea rows={4} placeholder='{"weather":"rain","time":"night"}' />
                                </Form.Item>
                                <Button
                                    htmlType='submit'
                                    type='primary'
                                    loading={loading}
                                    disabled={Boolean(overview.dataset)}
                                >
                                    {overview.dataset ? 'Ground Truth Job already exists' : 'Create Ground Truth Job'}
                                </Button>
                                {overview.dataset && (
                                    <Typography.Text type='secondary' style={{ marginLeft: 12 }}>
                                        Delete or reset the existing Ground Truth configuration before creating another one.
                                    </Typography.Text>
                                )}
                            </Form>
                        </Card>
                    </Col>
                    <Col xs={24} lg={12}>
                        <Card title='Phase 2 — YOLO Evaluation'>
                            <Alert
                                type='info'
                                showIcon
                                message='Paths are resolved inside the CVAT backend container.'
                            />
                            <Form
                                layout='vertical'
                                onFinish={(values) => submit(`${endpoint}/runs`, values)}
                                initialValues={{ confidence: 0.25, iou_threshold: 0.5 }}
                            >
                                <Form.Item name='model_path' label='YOLO model path' rules={[{ required: true }]}>
                                    <Input placeholder='/opt/cvat/evaluation-data/models/best.pt' />
                                </Form.Item>
                                <Space>
                                    <Form.Item name='confidence' label='Confidence'>
                                        <InputNumber min={0} max={1} step={0.05} />
                                    </Form.Item>
                                    <Form.Item name='iou_threshold' label='IoU'>
                                        <InputNumber min={0} max={1} step={0.05} />
                                    </Form.Item>
                                </Space>
                                <Button
                                    htmlType='submit'
                                    type='primary'
                                    loading={loading}
                                    disabled={!canEvaluate}
                                >
                                    Run evaluation on Ground Truth
                                </Button>
                            </Form>
                        </Card>
                    </Col>
                </Row>
                {overview.dataset && validationIssues.length > 0 && (
                    <Card title='Ground Truth validation issues' className='cvat-model-assisted-qc-section'>
                        <Table
                            rowKey={(issue, index) => `${issue.frame}-${issue.rule}-${issue.shape_id || index}`}
                            dataSource={validationIssues}
                            pagination={{ pageSize: 20 }}
                            columns={[
                                {
                                    title: 'Severity',
                                    dataIndex: 'severity',
                                    render: (value) => <Tag color={value === 'blocker' ? 'red' : 'orange'}>{value}</Tag>,
                                },
                                { title: 'Rule', dataIndex: 'rule' },
                                { title: 'Message', dataIndex: 'message' },
                                {
                                    title: 'Frame',
                                    dataIndex: 'frame',
                                    render: (frame) => (
                                        <Button
                                            type='link'
                                            onClick={() => history.push(
                                                `/tasks/${tid}/jobs/${overview.dataset?.ground_truth_job_id}?frame=${frame}`,
                                            )}
                                        >
                                            Open frame {frame}
                                        </Button>
                                    ),
                                },
                            ]}
                        />
                    </Card>
                )}
                <Card
                    title='Phase 3 — Condition & Threshold Analysis'
                    className='cvat-model-assisted-qc-section'
                    style={{ display: 'none' }}
                >
                    <Alert
                        type='info'
                        showIcon
                        message='Runs every Confidence × IoU combination on the validated Ground Truth dataset.'
                    />
                    <Form
                        layout='vertical'
                        initialValues={{
                            model_path: '/opt/cvat/evaluation-data/models/yolov8x.pt',
                            confidence_values: '0.15,0.25,0.35',
                            iou_values: '0.5,0.7',
                            objective: 'map50_95',
                        }}
                        onFinish={(values) => submit(`${endpoint}/threshold-analyses`, {
                            ...values,
                            confidence_values: parseThresholds(values.confidence_values),
                            iou_values: parseThresholds(values.iou_values),
                        })}
                    >
                        <Row gutter={16}>
                            <Col xs={24} lg={8}>
                                <Form.Item name='model_path' label='YOLO model path' rules={[{ required: true }]}>
                                    <Input />
                                </Form.Item>
                            </Col>
                            <Col xs={24} md={8} lg={5}>
                                <Form.Item
                                    name='confidence_values'
                                    label='Confidence values'
                                    rules={[{ required: true }]}
                                >
                                    <Input placeholder='0.15,0.25,0.35' />
                                </Form.Item>
                            </Col>
                            <Col xs={24} md={8} lg={5}>
                                <Form.Item name='iou_values' label='IoU values' rules={[{ required: true }]}>
                                    <Input placeholder='0.5,0.7' />
                                </Form.Item>
                            </Col>
                            <Col xs={24} md={8} lg={6}>
                                <Form.Item name='objective' label='Optimization objective'>
                                    <Select options={[
                                        { value: 'map50_95', label: 'mAP50–95' },
                                        { value: 'map50', label: 'mAP50' },
                                        { value: 'precision', label: 'Precision' },
                                        { value: 'recall', label: 'Recall' },
                                        { value: 'f1', label: 'F1 score' },
                                    ]} />
                                </Form.Item>
                            </Col>
                        </Row>
                        <Space>
                            <Button htmlType='submit' type='primary' loading={loading} disabled={!canEvaluate}>
                                Run threshold analysis
                            </Button>
                            <Typography.Text>
                                Conditions: {JSON.stringify(overview.dataset?.conditions || {})}
                            </Typography.Text>
                        </Space>
                    </Form>
                    {latestAnalysis?.status === 'completed' && (
                        <Alert
                            type='success'
                            showIcon
                            message={bestSummary}
                        />
                    )}
                    {latestAnalysis?.error && <Alert type='error' showIcon message={latestAnalysis.error} />}
                    <Table
                        rowKey={(result) => `${result.confidence}-${result.iou_threshold}`}
                        dataSource={latestAnalysis?.results || []}
                        pagination={false}
                        columns={[
                            { title: 'Confidence', dataIndex: 'confidence' },
                            { title: 'IoU', dataIndex: 'iou_threshold' },
                            { title: 'mAP50', dataIndex: 'map50', render: (value) => value.toFixed(4) },
                            { title: 'mAP50–95', dataIndex: 'map50_95', render: (value) => value.toFixed(4) },
                            { title: 'Precision', dataIndex: 'precision', render: (value) => value.toFixed(4) },
                            { title: 'Recall', dataIndex: 'recall', render: (value) => value.toFixed(4) },
                            { title: 'F1', dataIndex: 'f1', render: (value) => value.toFixed(4) },
                        ]}
                    />
                    <Typography.Title level={4}>Performance by environment condition</Typography.Title>
                    <Table
                        rowKey={(summary) => JSON.stringify(summary.conditions)}
                        dataSource={conditionSummaries}
                        pagination={false}
                        columns={[
                            {
                                title: 'Conditions',
                                dataIndex: 'conditions',
                                render: (conditions) => JSON.stringify(conditions),
                            },
                            { title: 'Runs', dataIndex: 'runs' },
                            { title: 'Avg mAP50', dataIndex: 'map50', render: (value) => value.toFixed(4) },
                            {
                                title: 'Avg mAP50–95',
                                dataIndex: 'map50_95',
                                render: (value) => value.toFixed(4),
                            },
                            {
                                title: 'Avg Precision',
                                dataIndex: 'precision',
                                render: (value) => value.toFixed(4),
                            },
                            { title: 'Avg Recall', dataIndex: 'recall', render: (value) => value.toFixed(4) },
                            { title: 'Avg F1', dataIndex: 'f1', render: (value) => value.toFixed(4) },
                        ]}
                    />
                </Card>
                <Card
                    title='Phase 4 — Risk Scoring & Review Prioritization'
                    className='cvat-model-assisted-qc-section'
                    style={{ display: 'none' }}
                >
                    <Alert
                        type='info'
                        showIcon
                        message='Creates idempotent, explainable alerts from saved predictions and verified GT.'
                    />
                    <Form
                        layout='inline'
                        initialValues={{ match_iou: 0.5, min_confidence: 0.5, dedup_iou: 0.9 }}
                        onFinish={(values) => submit(`${endpoint}/alerts`, {
                            ...values,
                            evaluation_run_id: latestCompletedRun?.id,
                            condition_reliability: 1,
                        })}
                    >
                        <Form.Item name='match_iou' label='Match IoU'><InputNumber min={0.01} max={1} step={0.05} /></Form.Item>
                        <Form.Item name='min_confidence' label='Min confidence'>
                            <InputNumber min={0} max={1} step={0.05} />
                        </Form.Item>
                        <Form.Item name='dedup_iou' label='Dedup IoU'><InputNumber min={0.01} max={1} step={0.05} /></Form.Item>
                        <Button htmlType='submit' type='primary' loading={loading} disabled={!latestCompletedRun}>
                            Generate review queue
                        </Button>
                    </Form>
                </Card>
                <Card
                    title='Phase 5 — Human Review'
                    className='cvat-model-assisted-qc-section'
                    style={{ display: 'none' }}
                >
                    <Alert
                        type='warning'
                        showIcon
                        message='Review decisions never modify CVAT annotations automatically.'
                    />
                    <Table
                        rowKey='id'
                        dataSource={alerts}
                        pagination={{ pageSize: 20 }}
                        columns={[
                            { title: 'Risk', dataIndex: 'risk_score', render: (value) => value.toFixed(4) },
                            { title: 'Type', dataIndex: 'type' },
                            {
                                title: 'Classes',
                                render: (_, alert) => `${alert.prediction.class_name} / ${alert.annotation?.class_name || 'none'}`,
                            },
                            { title: 'Confidence', render: (_, alert) => alert.prediction.confidence.toFixed(4) },
                            {
                                title: 'Frame',
                                render: (_, alert) => (
                                    <Button
                                        type='link'
                                        onClick={() => history.push(
                                            `/tasks/${tid}/jobs/${overview.dataset?.ground_truth_job_id}?frame=${alert.frame}`,
                                        )}
                                    >
                                        {alert.frame}
                                    </Button>
                                ),
                            },
                            { title: 'Status', dataIndex: 'status', render: (value) => <Tag>{value}</Tag> },
                            {
                                title: 'Review',
                                render: (_, alert) => (
                                    <Space>
                                        {['PENDING', 'IN_REVIEW'].includes(alert.status) && (
                                            <>
                                                <Button onClick={() => submit(
                                                    `/api/model-assisted-qc/alerts/${alert.id}/review`,
                                                    { decision: 'CONFIRMED' },
                                                )}>Confirm</Button>
                                                <Button danger onClick={() => submit(
                                                    `/api/model-assisted-qc/alerts/${alert.id}/review`,
                                                    { decision: 'REJECTED' },
                                                )}>Reject</Button>
                                            </>
                                        )}
                                        {alert.status === 'CONFIRMED' && (
                                            <Button onClick={() => submit(
                                                `/api/model-assisted-qc/alerts/${alert.id}/review`,
                                                { decision: 'RESOLVED' },
                                            )}>Mark resolved</Button>
                                        )}
                                    </Space>
                                ),
                            },
                        ]}
                    />
                </Card>
                <Card
                    title='Phase 6 — QC Benchmark & Reporting'
                    className='cvat-model-assisted-qc-section'
                    style={{ display: 'none' }}
                >
                    <Form
                        layout='vertical'
                        initialValues={{ top_k: '20,50', seed: 42, verified_errors: '[]' }}
                        onFinish={(values) => {
                            try {
                                submit(`${endpoint}/benchmarks`, {
                                    evaluation_run_id: latestCompletedRun?.id,
                                    seed: values.seed,
                                    top_k: parseThresholds(values.top_k),
                                    verified_errors: JSON.parse(values.verified_errors),
                                });
                            } catch (error: any) {
                                notification.error({ message: 'Invalid verified-error JSON', description: error.message });
                            }
                        }}
                    >
                        <Row gutter={16}>
                            <Col span={6}><Form.Item name='seed' label='Seed'><InputNumber min={0} /></Form.Item></Col>
                            <Col span={6}><Form.Item name='top_k' label='Top K'><Input placeholder='20,50' /></Form.Item></Col>
                            <Col span={12}>
                                <Form.Item name='verified_errors' label='Verified errors JSON'>
                                    <Input.TextArea rows={3} />
                                </Form.Item>
                            </Col>
                        </Row>
                        <Button htmlType='submit' type='primary' disabled={!latestCompletedRun}>Run QC benchmark</Button>
                    </Form>
                    {latestBenchmark && (
                        <Alert
                            type='success'
                            showIcon
                            message={`Ranked QC Precision ${latestBenchmark.qc_precision.toFixed(4)}, QC Recall ${latestBenchmark.qc_recall.toFixed(4)}`}
                        />
                    )}
                </Card>
                <Alert
                    className='cvat-model-assisted-qc-section'
                    type='info'
                    showIcon
                    message='Phases 3–6 are integrated into the annotation workspace'
                    description='Open a regular annotation job and select QC Assistant in the top toolbar to detect, prioritize, review, and resolve suspected missing-object or wrong-class labels.'
                />
                <Card title='YOLO evaluation history' className='cvat-model-assisted-qc-section'>
                    <Table rowKey='id' dataSource={overview.runs} pagination={false} columns={[
                        { title: 'Run', dataIndex: 'id', render: (id) => `#${id}` },
                        { title: 'Model', dataIndex: 'model_path' },
                        {
                            title: 'Status',
                            dataIndex: 'status',
                            render: (value) => <Tag color={value === 'completed' ? 'green' : 'red'}>{value}</Tag>,
                        },
                        { title: 'mAP50', render: (_, run) => run.metrics?.map50?.toFixed(4) || '—' },
                        { title: 'Precision', render: (_, run) => run.metrics?.precision?.toFixed(4) || '—' },
                        { title: 'Recall', render: (_, run) => run.metrics?.recall?.toFixed(4) || '—' },
                        { title: 'Error', dataIndex: 'error' },
                    ]} />
                </Card>
            </Col>
        </Row>
    );
}
