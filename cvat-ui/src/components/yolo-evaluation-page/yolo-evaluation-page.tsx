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
    dataset: null | { ground_truth_job_id: number; sampled_frames: number[]; conditions: object };
    runs: EvaluationRun[];
}

export default function YoloEvaluationPage(): JSX.Element {
    const { tid } = useParams<{ tid: string }>();
    const history = useHistory();
    const [overview, setOverview] = useState<Overview>({ dataset: null, runs: [] });
    const [loading, setLoading] = useState(false);
    const endpoint = `/api/yolo-evaluation/tasks/${tid}`;
    const load = (): Promise<void> => Axios.get(`${endpoint}/dataset`).then(({ data }) => setOverview(data));

    useEffect(() => { load().catch(() => undefined); }, [tid]);

    const submit = async (url: string, values: Record<string, any>): Promise<void> => {
        setLoading(true);
        try {
            await Axios.post(url, values);
            notification.success({ message: 'Operation completed' });
            await load();
        } catch (error: any) {
            notification.error({
                message: 'Operation failed',
                description: error.response?.data?.error || error.message,
            });
            await load();
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

    return (
        <Row justify='center' className='cvat-yolo-evaluation-page'>
            <Col span={22} xl={18}>
                <Button
                    type='link'
                    icon={<LeftOutlined />}
                    onClick={() => history.push(`/tasks/${tid}`)}
                >
                    Back to task
                </Button>
                <Typography.Title level={2}><ExperimentOutlined /> Dataset & YOLO Evaluation</Typography.Title>
                <Row gutter={[16, 16]}>
                    <Col xs={24} lg={12}>
                        <Card title='Phase 1 — Dataset & Ground Truth'>
                            {overview.dataset && (
                                <Alert
                                    type='success'
                                    showIcon
                                    message={`GT Job #${overview.dataset.ground_truth_job_id} — ${overview.dataset.sampled_frames.length} frames`}
                                />
                            )}
                            <Form
                                layout='vertical'
                                onFinish={createDataset}
                                initialValues={{
                                    sampling_method: 'random_uniform', frame_count: 100, random_seed: 0, conditions: '{}',
                                }}
                            >
                                <Form.Item name='sampling_method' label='Sampling method'>
                                    <Select options={[
                                        { value: 'random_uniform', label: 'Random uniform' },
                                        { value: 'manual', label: 'Manual frame list' },
                                    ]} />
                                </Form.Item>
                                <Form.Item name='frame_count' label='Number of frames'>
                                    <InputNumber min={1} />
                                </Form.Item>
                                <Form.Item
                                    name='frames'
                                    label='Manual frames (comma-separated)'
                                    getValueFromEvent={(event) => event.target.value.split(',').filter(Boolean).map(Number)}
                                >
                                    <Input placeholder='0, 10, 25' />
                                </Form.Item>
                                <Form.Item name='random_seed' label='Random seed'><InputNumber min={0} /></Form.Item>
                                <Form.Item name='conditions' label='Environment metadata (JSON)'>
                                    <Input.TextArea rows={4} placeholder='{"weather":"rain","time":"night"}' />
                                </Form.Item>
                                <Button
                                    htmlType='submit'
                                    type='primary'
                                    loading={loading}
                                    disabled={Boolean(overview.dataset)}
                                >
                                    Create Ground Truth dataset
                                </Button>
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
                                    disabled={!overview.dataset}
                                >
                                    Run evaluation on Ground Truth
                                </Button>
                            </Form>
                        </Card>
                    </Col>
                </Row>
                <Card title='Evaluation history' className='cvat-yolo-evaluation-history'>
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
