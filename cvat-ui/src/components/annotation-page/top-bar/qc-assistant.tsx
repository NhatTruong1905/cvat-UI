import React, { useEffect, useMemo, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import Axios from 'axios';
import { AuditOutlined, ExclamationCircleOutlined } from '@ant-design/icons';
import Alert from 'antd/lib/alert';
import Button from 'antd/lib/button';
import Input from 'antd/lib/input';
import Modal from 'antd/lib/modal';
import notification from 'antd/lib/notification';
import Progress from 'antd/lib/progress';
import Space from 'antd/lib/space';
import Table from 'antd/lib/table';
import Tabs from 'antd/lib/tabs';
import Tag from 'antd/lib/tag';
import Typography from 'antd/lib/typography';

import { Canvas } from 'cvat-canvas-wrapper';
import { Job, ObjectState } from 'cvat-core-wrapper';
import { activateObject } from 'actions/annotation-actions';
import { CombinedState } from 'reducers';

interface QCAlert {
    id: number;
    frame: number;
    type: 'POSSIBLE_MISSING_OBJECT' | 'POSSIBLE_WRONG_CLASS' | 'POSSIBLE_EXTRA_OBJECT';
    prediction: { class_name: string; confidence: number; bbox_xyxy: number[] };
    annotation: null | {
        annotation_id?: number;
        track_id?: number;
        class_name: string;
        bbox_xyxy: number[];
    };
    iou: null | number;
    risk_score: number;
    reason_codes: string[];
    status: 'PENDING' | 'IN_REVIEW' | 'CONFIRMED' | 'REJECTED' | 'RESOLVED';
}

interface EvaluationRun {
    id: number;
    status: string;
    metrics: Record<string, number>;
}

interface Benchmark {
    id: number;
    metrics: {
        baselines?: Record<string, {
            qc_precision: number;
            qc_recall: number;
            precision_at_k: Record<string, number>;
        }>;
    };
}

interface Props {
    jobInstance: Job;
    currentFrame: number;
    onSelectFrame(frame: number): void;
}

const QC_OVERLAY_ID = 'cvat-qc-alert-overlay';

function showAlertsOverlay(canvas: Canvas, alerts: QCAlert[]): void {
    const content = canvas.html().querySelector('#cvat_canvas_content');
    if (!(content instanceof SVGSVGElement)) return;

    content.querySelector(`#${QC_OVERLAY_ID}`)?.remove();
    const namespace = 'http://www.w3.org/2000/svg';
    const group = document.createElementNS(namespace, 'g');
    group.id = QC_OVERLAY_ID;
    group.setAttribute('pointer-events', 'none');

    alerts.forEach((alert) => {
        const points = alert.annotation?.bbox_xyxy || alert.prediction.bbox_xyxy;
        if (!points || points.length !== 4) return;
        const [left, top, right, bottom] = points.map((value) => value + canvas.geometry.offset);

        const rectangle = document.createElementNS(namespace, 'rect');
        rectangle.setAttribute('x', String(left));
        rectangle.setAttribute('y', String(top));
        rectangle.setAttribute('width', String(right - left));
        rectangle.setAttribute('height', String(bottom - top));
        rectangle.setAttribute('fill', 'rgba(255, 0, 0, 0.14)');
        rectangle.setAttribute('stroke', '#ff0000');
        rectangle.setAttribute('stroke-width', '5');
        rectangle.setAttribute('stroke-dasharray', '12 7');
        rectangle.setAttribute('vector-effect', 'non-scaling-stroke');

        const label = document.createElementNS(namespace, 'text');
        label.setAttribute('x', String(left));
        label.setAttribute('y', String(Math.max(20, top - 8)));
        label.setAttribute('fill', '#ff0000');
        label.setAttribute('font-size', '20');
        label.setAttribute('font-weight', 'bold');
        label.setAttribute('paint-order', 'stroke');
        label.setAttribute('stroke', 'white');
        label.setAttribute('stroke-width', '4');
        label.setAttribute('vector-effect', 'non-scaling-stroke');
        if (alert.type === 'POSSIBLE_MISSING_OBJECT') {
            label.textContent = `QC: missing ${alert.prediction.class_name}`;
        } else if (alert.type === 'POSSIBLE_EXTRA_OBJECT') {
            label.textContent = `QC: verify human box (${alert.annotation?.class_name})`;
        } else {
            label.textContent = `QC: wrong class (${alert.annotation?.class_name} -> ${alert.prediction.class_name})`;
        }

        group.append(rectangle, label);
    });
    content.append(group);
}

function showAlertOverlay(canvas: Canvas, alert: QCAlert): void {
    showAlertsOverlay(canvas, [alert]);
}

function QCAssistant(props: Props): JSX.Element {
    const { jobInstance, currentFrame, onSelectFrame } = props;
    const [open, setOpen] = useState(false);
    const [loading, setLoading] = useState(false);
    const [alerts, setAlerts] = useState<QCAlert[]>([]);
    const [runs, setRuns] = useState<EvaluationRun[]>([]);
    const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
    const [frameQuery, setFrameQuery] = useState('');
    const [searchedFrame, setSearchedFrame] = useState<number | null>(null);
    const [pendingAlert, setPendingAlert] = useState<QCAlert | null>(null);
    const dispatch = useDispatch();
    const canvasInstance = useSelector(
        (state: CombinedState) => state.annotation.canvas.instance,
    );
    const annotationStates = useSelector(
        (state: CombinedState) => state.annotation.annotations.states as ObjectState[],
    );
    const frameFetching = useSelector(
        (state: CombinedState) => state.annotation.player.frame.fetching,
    );
    const taskID = jobInstance.taskId;
    const endpoint = `/api/model-assisted-qc/tasks/${taskID}`;

    const load = async (): Promise<void> => {
        if (!taskID) return;
        const [overviewResponse, alertsResponse, benchmarkResponse] = await Promise.all([
            Axios.get(`${endpoint}/dataset`),
            Axios.get(`${endpoint}/alerts`, { params: { job_id: jobInstance.id } }),
            Axios.get(`${endpoint}/benchmarks`),
        ]);
        setRuns(overviewResponse.data.runs || []);
        setAlerts(alertsResponse.data || []);
        setBenchmarks(benchmarkResponse.data || []);
    };

    useEffect(() => {
        if (open) load().catch((error) => notification.error({
            message: 'Could not load QC Assistant',
            description: error.response?.data?.error || error.message,
        }));
    }, [open, taskID]);

    useEffect(() => {
        if (!pendingAlert || currentFrame !== pendingAlert.frame || frameFetching) return;

        if (pendingAlert.annotation) {
            const serverID = pendingAlert.annotation.track_id ?? pendingAlert.annotation.annotation_id;
            const objectState = annotationStates.find((state) => state.serverID === serverID);
            if (objectState && canvasInstance instanceof Canvas) {
                dispatch(activateObject(objectState.clientID, null, null));
                window.requestAnimationFrame(() => canvasInstance.focus(objectState.clientID, 80));
            } else {
                notification.info({
                    message: 'Frame opened',
                    description: 'The matching Ground Truth box is not present in this annotation job.',
                });
            }
        } else {
            const coordinates = pendingAlert.prediction.bbox_xyxy.map((value) => Math.round(value)).join(', ');
            notification.info({
                message: 'Missing-object candidate opened',
                description: `The red box is YOLO's candidate at [${coordinates}]. It is not saved as an annotation.`,
            });
        }
        if (canvasInstance instanceof Canvas) showAlertOverlay(canvasInstance, pendingAlert);
        setPendingAlert(null);
    }, [pendingAlert, currentFrame, frameFetching, annotationStates, canvasInstance, dispatch]);

    const actionableAlerts = useMemo(
        () => alerts.filter((alert) => !['REJECTED', 'RESOLVED'].includes(alert.status)),
        [alerts],
    );
    const currentFrameAlerts = useMemo(
        () => actionableAlerts.filter((alert) => alert.frame === currentFrame),
        [actionableAlerts, currentFrame],
    );
    const visibleAlerts = searchedFrame === null ? actionableAlerts : actionableAlerts.filter(
        (alert) => alert.frame === searchedFrame,
    );
    const completedRun = runs.find((run) => run.status === 'completed');
    const latestBenchmark = benchmarks[0]?.metrics?.baselines?.risk_ranked;

    useEffect(() => {
        if (
            open &&
            searchedFrame === currentFrame &&
            canvasInstance instanceof Canvas
        ) {
            showAlertsOverlay(canvasInstance, currentFrameAlerts);
        }
    }, [open, searchedFrame, currentFrame, currentFrameAlerts, canvasInstance]);

    const runQC = async (): Promise<void> => {
        if (!completedRun) {
            notification.warning({
                message: 'YOLO evaluation is required',
                description: 'Complete Phase 2 for this task before running QC Assistant.',
            });
            return;
        }
        setLoading(true);
        try {
            await Axios.post(`${endpoint}/alerts`, {
                evaluation_run_id: completedRun.id,
                job_id: jobInstance.id,
                match_iou: 0.5,
                min_confidence: 0.5,
                dedup_iou: 0.9,
                condition_reliability: 1,
            });
            await load();
            notification.success({ message: 'QC review queue is ready' });
        } catch (error: any) {
            notification.error({
                message: 'QC detection failed',
                description: error.response?.data?.error || error.message,
            });
        } finally {
            setLoading(false);
        }
    };

    const review = async (alertID: number, decision: string): Promise<void> => {
        setLoading(true);
        try {
            await Axios.post(`/api/model-assisted-qc/alerts/${alertID}/review`, { decision });
            await load();
        } catch (error: any) {
            notification.error({
                message: 'Review action failed',
                description: error.response?.data?.error || error.message,
            });
        } finally {
            setLoading(false);
        }
    };

    const navigateToAlert = (alert: QCAlert): void => {
        setPendingAlert(alert);
        onSelectFrame(alert.frame);
        setOpen(false);
    };

    const openCurrentFrameErrors = (): void => {
        setFrameQuery(String(currentFrame));
        setSearchedFrame(currentFrame);
        if (canvasInstance instanceof Canvas) {
            showAlertsOverlay(canvasInstance, currentFrameAlerts);
        }
        setOpen(true);
    };

    const openQCAssistant = (): void => {
        setFrameQuery('');
        setSearchedFrame(null);
        setOpen(true);
    };

    const searchFrame = (value: string): void => {
        const match = value.trim().match(/^(?:frame\s*)?#?(\d+)$/i);
        if (!match) {
            notification.warning({
                message: 'Invalid frame number',
                description: 'Enter a frame number, for example: 1, #1, or Frame 1.',
            });
            return;
        }
        setFrameQuery(match[1]);
        setSearchedFrame(Number(match[1]));
    };

    const clearFrameSearch = (): void => {
        setFrameQuery('');
        setSearchedFrame(null);
    };

    const columns = [
        {
            title: 'Risk',
            dataIndex: 'risk_score',
            width: 95,
            render: (value: number): JSX.Element => <Progress percent={Math.round(value * 100)} size='small' />,
        },
        {
            title: 'Suspected issue',
            render: (_: unknown, alert: QCAlert): JSX.Element => (
                <Space direction='vertical' size={0}>
                    <Tag color={alert.type === 'POSSIBLE_MISSING_OBJECT' ? 'orange' :
                        alert.type === 'POSSIBLE_EXTRA_OBJECT' ? 'red' : 'purple'}>
                        {alert.type === 'POSSIBLE_MISSING_OBJECT' ? 'Missing object' :
                            alert.type === 'POSSIBLE_EXTRA_OBJECT' ? 'Verify human box' : 'Wrong class'}
                    </Tag>
                    {alert.type === 'POSSIBLE_EXTRA_OBJECT' && (
                        <Typography.Text type='secondary'>
                            Human label: {alert.annotation?.class_name}; no matching YOLO object
                        </Typography.Text>
                    )}
                    <Typography.Text type='secondary'>
                        Model: {alert.prediction.class_name} ({alert.prediction.confidence.toFixed(2)})
                        {alert.annotation ? ` · Label: ${alert.annotation.class_name}` : ''}
                    </Typography.Text>
                </Space>
            ),
        },
        {
            title: 'Open error',
            dataIndex: 'frame',
            width: 120,
            render: (frame: number, alert: QCAlert): JSX.Element => (
                <Button
                    type={frame === currentFrame ? 'primary' : 'link'}
                    onClick={() => navigateToAlert(alert)}
                >
                    Frame {frame}
                </Button>
            ),
        },
        {
            title: 'Status',
            dataIndex: 'status',
            width: 110,
            render: (value: string): JSX.Element => <Tag>{value}</Tag>,
        },
        {
            title: 'Reviewer decision',
            width: 240,
            render: (_: unknown, alert: QCAlert): JSX.Element => (
                <Space>
                    {['PENDING', 'IN_REVIEW'].includes(alert.status) && (
                        <>
                            <Button size='small' onClick={() => review(alert.id, 'CONFIRMED')}>Confirm</Button>
                            <Button size='small' danger onClick={() => review(alert.id, 'REJECTED')}>Reject</Button>
                        </>
                    )}
                    {alert.status === 'CONFIRMED' && (
                        <Button size='small' onClick={() => review(alert.id, 'RESOLVED')}>Resolved</Button>
                    )}
                </Space>
            ),
        },
    ];

    const tabItems = [
        {
            key: 'queue',
            label: `Review queue (${actionableAlerts.length})`,
            children: (
                <Space direction='vertical' style={{ width: '100%' }}>
                    {currentFrameAlerts.length > 0 && (
                        <Alert
                            type='warning'
                            showIcon
                            message={`${currentFrameAlerts.length} QC alert(s) on the current frame`}
                        />
                    )}
                    <Space>
                        <Button type='primary' loading={loading} onClick={runQC}>Run / refresh QC detection</Button>
                        <Typography.Text type='secondary'>
                            Uses the latest completed YOLO evaluation with safe default QC thresholds.
                        </Typography.Text>
                    </Space>
                    <Space style={{ width: '100%' }}>
                        <Input.Search
                            value={frameQuery}
                            allowClear
                            placeholder='Search frame, e.g. 1 or Frame 1'
                            enterButton='Search frame'
                            style={{ maxWidth: 420 }}
                            onChange={(event) => {
                                setFrameQuery(event.target.value);
                                if (!event.target.value) setSearchedFrame(null);
                            }}
                            onSearch={searchFrame}
                        />
                        {searchedFrame !== null && (
                            <Button onClick={clearFrameSearch}>Show all frames</Button>
                        )}
                    </Space>
                    {searchedFrame !== null && (
                        <Alert
                            type={visibleAlerts.length ? 'success' : 'info'}
                            showIcon
                            message={visibleAlerts.length ?
                                `${visibleAlerts.length} QC alert(s) found on frame ${searchedFrame}` :
                                `No active QC alerts found on frame ${searchedFrame}`}
                        />
                    )}
                    <Table
                        rowKey='id'
                        size='small'
                        loading={loading}
                        dataSource={visibleAlerts}
                        columns={columns}
                        pagination={{ pageSize: 8, hideOnSinglePage: true }}
                    />
                </Space>
            ),
        },
        {
            key: 'metrics',
            label: 'Quality metrics',
            children: (
                <Space direction='vertical' style={{ width: '100%' }}>
                    {completedRun ? (
                        <Alert
                            type='info'
                            showIcon
                            message={`YOLO run #${completedRun.id}`}
                            description={`mAP50 ${(completedRun.metrics?.map50 || 0).toFixed(4)} · Precision ${(completedRun.metrics?.precision || 0).toFixed(4)} · Recall ${(completedRun.metrics?.recall || 0).toFixed(4)}`}
                        />
                    ) : <Alert type='warning' message='No completed YOLO evaluation is available.' />}
                    {latestBenchmark ? (
                        <Alert
                            type='success'
                            showIcon
                            message='Latest risk-ranked QC benchmark'
                            description={`QC Precision ${latestBenchmark.qc_precision.toFixed(4)} · QC Recall ${latestBenchmark.qc_recall.toFixed(4)} · Precision@K ${JSON.stringify(latestBenchmark.precision_at_k)}`}
                        />
                    ) : (
                        <Alert
                            type='info'
                            message='No verified QC benchmark yet'
                            description='Benchmarking requires an independently verified error manifest.'
                        />
                    )}
                </Space>
            ),
        },
    ];

    return (
        <>
            <Button
                type={currentFrameAlerts.length ? 'primary' : 'link'}
                className='cvat-annotation-header-button'
                icon={<AuditOutlined />}
                onClick={openQCAssistant}
            >
                QC Assistant{actionableAlerts.length ? ` (${actionableAlerts.length})` : ''}
            </Button>
            <Button
                type={currentFrameAlerts.length ? 'primary' : 'link'}
                danger={currentFrameAlerts.length > 0}
                className='cvat-annotation-header-button'
                icon={<ExclamationCircleOutlined />}
                onClick={openCurrentFrameErrors}
            >
                Frame Errors ({currentFrameAlerts.length})
            </Button>
            <Modal
                title={searchedFrame === null ? 'Model-assisted QC Assistant' : `Frame ${searchedFrame} errors`}
                open={open}
                width={1100}
                onCancel={() => setOpen(false)}
                footer={<Button onClick={() => setOpen(false)}>Close</Button>}
            >
                <Alert
                    type='info'
                    showIcon
                    message='YOLO proposes suspicious annotations; the reviewer makes the final decision.'
                    description='Confirming an alert never changes CVAT annotations automatically. Correct the label manually, save it, then mark the alert Resolved.'
                    style={{ marginBottom: 16 }}
                />
                <Tabs items={tabItems} />
            </Modal>
        </>
    );
}

export default React.memo(QCAssistant);
