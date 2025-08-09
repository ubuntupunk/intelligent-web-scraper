"""
Unit tests for the performance monitoring system.

This module tests the comprehensive performance monitoring capabilities including
response time tracking, memory usage analysis, throughput measurement, and
performance benchmarking.
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from intelligent_web_scraper.monitoring.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetric,
    PerformanceBenchmark,
    PerformanceOptimizationReport,
    OperationTracker
)


class TestPerformanceMetric:
    """Test PerformanceMetric data class."""
    
    def test_performance_metric_creation(self):
        """Test creating a performance metric."""
        metric = PerformanceMetric(
            operation_type="test_operation",
            operation_id="test_123",
            response_time_ms=1500.0,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            success=True
        )
        
        assert metric.operation_type == "test_operation"
        assert metric.operation_id == "test_123"
        assert metric.response_time_ms == 1500.0
        assert metric.memory_usage_mb == 256.0
        assert metric.cpu_usage_percent == 45.0
        assert metric.success is True
        assert metric.error_message is None
    
    def test_performance_metric_to_dict(self):
        """Test converting performance metric to dictionary."""
        metric = PerformanceMetric(
            operation_type="scraping",
            operation_id="scrape_001",
            response_time_ms=2000.0,
            success=False,
            error_message="Connection timeout"
        )
        
        result = metric.to_dict()
        
        assert result['operation_type'] == "scraping"
        assert result['operation_id'] == "scrape_001"
        assert result['response_time_ms'] == 2000.0
        assert result['success'] is False
        assert result['error_message'] == "Connection timeout"
        assert 'timestamp' in result


class TestPerformanceBenchmark:
    """Test PerformanceBenchmark data class."""
    
    def test_benchmark_creation(self):
        """Test creating a performance benchmark."""
        benchmark = PerformanceBenchmark(
            benchmark_name="test_benchmark",
            total_operations=100,
            successful_operations=95,
            failed_operations=5,
            average_response_time_ms=1200.0,
            throughput_ops_per_sec=8.5
        )
        
        assert benchmark.benchmark_name == "test_benchmark"
        assert benchmark.total_operations == 100
        assert benchmark.successful_operations == 95
        assert benchmark.failed_operations == 5
        assert benchmark.average_response_time_ms == 1200.0
        assert benchmark.throughput_ops_per_sec == 8.5
    
    def test_benchmark_to_dict(self):
        """Test converting benchmark to dictionary."""
        benchmark = PerformanceBenchmark(
            benchmark_name="api_benchmark",
            total_operations=50,
            throughput_ops_per_sec=12.3
        )
        
        result = benchmark.to_dict()
        
        assert result['benchmark_name'] == "api_benchmark"
        assert result['total_operations'] == 50
        assert result['throughput_ops_per_sec'] == 12.3
        assert 'timestamp' in result


class TestOperationTracker:
    """Test OperationTracker helper class."""
    
    def test_operation_tracker_creation(self):
        """Test creating an operation tracker."""
        tracker = OperationTracker("op_123", "test_operation")
        
        assert tracker.operation_id == "op_123"
        assert tracker.operation_type == "test_operation"
        assert tracker.success is True
        assert tracker.error_message is None
        assert tracker.metadata == {}
    
    def test_operation_tracker_set_success(self):
        """Test setting operation success status."""
        tracker = OperationTracker("op_456", "scraping")
        
        # Test success
        tracker.set_success(True)
        assert tracker.success is True
        assert tracker.error_message is None
        
        # Test failure
        tracker.set_success(False, "Network error")
        assert tracker.success is False
        assert tracker.error_message == "Network error"
    
    def test_operation_tracker_metadata(self):
        """Test adding metadata to operation tracker."""
        tracker = OperationTracker("op_789", "analysis")
        
        tracker.add_metadata("url", "https://example.com")
        tracker.add_metadata("items_found", 42)
        
        assert tracker.metadata["url"] == "https://example.com"
        assert tracker.metadata["items_found"] == 42


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""
    
    @pytest.fixture
    def monitor(self):
        """Create a performance monitor for testing."""
        return PerformanceMonitor(history_size=100, enable_detailed_tracking=True)
        assert monitor.history_size == 100
        assert monitor.enable_detailed_tracking is True
        
        assert len(monitor.operation_metrics) =0
        assert len(monitor.benchmark_results) == 0
        assert monitor.is_monitoring i
    
    def test_start_stop_monitoring(seor):
    "
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.is_monitoring is True
        one
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor.is_monitoring is False
    
    def test_record_performance
        """Test recording performance metrics."""
        metric = PerformanceMetric(
            operation_type="test_op",
    
            response_time_ms=1500.0,
            memory_usage_mb=256.0,
            success=True
        )
        
        monitor.record_performance_metriic)
        
        assert len(monitor.performance_metrics) == 1
        assert len(monitor.operation_metrics[
 1
    
    @patch('intelligent_web_s)
    def test_track_operation_context_manager(
    "
        # Mock proc
        mock_process_i
        mock_process_instance.memory_info.return_value. # 256 MB
        mock_process_instance.cpu_lue = 45.0
        mock_process.return_v
        
        # Use context manager
        w
     work
            tracker.set_success(True)
        
        # Check that metric was recorded
        assert len(monitor.performance_metrics) == 1
        metric = monitor.performance_metrics[0]
        assert metric.operation_type == "test
        assert metric.operation_id == "op_001"
        assert metric.success is True
    
    
    @patch('intelligent_web_scraper.monitoring.perfos')
    def test_track_operation_wi):
        """Test track_operation wi
        # Mock process metrics
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_va1024
        
        mock_process.return_vance
        
        # Use context manager with error
        er:
            tracker.set_success(Falor")
        
        # Check that metric was recorded with error
    ) == 1
        metric = monitor.performance_metrics[0]
        assert metric.success is False
        assert metric.error_messageor"
    
    def test_run_benchmark_sequentia:
        """Test running a sequential""
        def test_operation():
            time.sleep(0e work
         
        
        benchmark = monitor.run_benchmark(
        
            operation_func=test_operation,
            num_operations=5,
            concurrent_operations=1,
        ations=2
        )
        
        assert benchmark.benchmark_name == "test_benchmark"
    
        assert benchmark.successful_operations == 5
        assert benchmark.failed_operations == 0
        assert benchmark.success_rate_percent == 100.0
        assert benchmark.throughput_ops_per_sec > 0
        assert benchmark.average_respo
    
    def test_run_benchmark_concurrent(self, monitor):
        """Test running a concurrent benchmark."""
        :
            time.sleep(0.001)
            return "success"
        
        benchmark = monitor.run_bench(
        rk",
            operation_func=test_operation,
            num_operations=10,
            concurrent_operations=3,
            warmup_operations=0
        )
        
    "
        assert benchmark.total_operations == 10
        assert benchmark.throughput_ops_per_sec > 0
    
    def test_run_benchitor):
        """Test benchmark with some fa"
        call_count = 0
        
        def failing_operation():
        nt
            call_count += 1
            if call_count % 3 == 0:  # Every 3rd operation fails
                raise Exception("Test failure")
        ess"
        
        benchmark = monitor.run_benchmark(
            benchmark_name="failing_be,
            operation_func=failing_operation,
    ns=9,
            concurrent_operations=1,
            warmup_operations=0
        )
        
        assert benchmark.be
        
        assert benchmark.failed_operations= 3
        assert benchmark.successful_operatio= 6
        assert benchmark.error_rate_percen > 0
        assert benchmark.succ
    
    def test_calculate_percentiitor):
        """
        0.0]
        
        assert monitor._calculate_percentile(v Median
        assert monitor._calculate_percentile(values== 9.1
        assert monitor._calculate_percentile(va
        assert monitor._calculate_percentile(values= 9.91
        
        
        assert monitor._calculate_percent0
        assert monitor._calculate_percentile([], 5) == 0.0
    
    def test_performance_summary(self, monitor):
        """Test getting performance summary."""
        # Add some test metri
        for i in range(10):
            metric = PerformanceMetric(
        _test",
                operation_id=f"op_{i}",
                response_time_ms=1000.0 + i *00,
                memory_usage_mb=100.0 + i  10,
                cpu_usage_perc 5,
                success=i < 8  # 8 sled
            )
         ric)
        
        summary = monitor.get_performance_summary(hours=1.0)
        
        assert summary['total_operations'] == 10
    = 8
        assert summary['overall_stats']['success_rate'] 80.0
        assert 'summary_test' in summary['operation_types'
        assert summary 10
        .0
    
    def test_performance_summar
        """Test performance""
        summary = monitor.get_performance_summary(hours=1.0)
        
        assert summary['tota0
        {}
        assert summary['overall_stats'] == {}
    
    def test_generate_optimization_report(sel
        """Test generating oport."""
        # Add metrics with various p
        for i in range(20):
         Metric(
        ,
                operation_id=f"opt_{i}",
                response_time_ms=2000.0 + i * s
                memory_usage_mb=300.0 + i * 20,     # Increasing memorsage
                cpu_usage_percent=50.0 + i * 2,    sage
                success=i < 18  # Most successful
    
            monitor.record_performance_metric(met
        
        report = monitor.generate_optimization_report(analysis_hours)
        
        assert report.report_id.startswith("opt_report_")
        assert report.analysis_period_hours == 1.0
        assert report.overall_performance_score >= 0.0
        
        assert report.average_respo> 0
        assert len(report.optimization_recommendations) > 0
    
    def tor):
        """Test trend cal""
        # Increasing trend
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0
    g"
        
        # Decreasing trend
        decreasing_values = [10.0, 1.0]
        assert monitor._cal
        
        # Stable trend
        stable_values = [5.0, 5.1, 4.9,]
        assert monitor._calculate_trend(stable_val"
        
        # Edg
        assert monitor._calculate_trend([]) == "stable"
        
    
    def 
        """Test setting and using performance ba
        # Set baseline
        baseline_metrics = {
            'response_time_ms': 2000.0,
            'throughput_ops_per_sec': 5.0,
            'memory_usage_mb': 200.0
        }
        monitor.set_performance_baseline("test_operation", trics)
        
        # Add current metrics (better performance)
        for i in range(10):
            metric = PerformanceMetric(
        ,
                operation_id=f"baseline_test_{i
                response_time_ms=1500.0,  # Better than baseline
                throughput_ops_per_sec=7.0,  # Better t
                memory_usage_mb=150.0,  # Better than
    =True
            )
            monitor.record_performance_metric(metrc)
        
        comparison 
        
        assert "test_operation_response_t
        assert comparison["test_operation_response_time_int
        assert "test_operation_throughput_change" in co
        assert comparison["test_operation_throughputnt
    
    def test_p
        """Test performance ev"""
        callback_events = []
        
        def test_callback(event_data):
            callback_events.append(event_data)
        
        monitollback)
        
        # Record a metric to trigger callbac
        metric = PerformanceMetric(
            operation_type="ca",
            operation_id="callback_001",
            r,
         
        )
        monitor.record_performtric)
        
        1
        assert callback_events[0]['type'] == 'metric_recorded'
        _test'
    
    def test_export_performance_data(self, monitor
        """Test exporting performance data."""
        # Add some test data
        for i in range(5):
            metric = PerformanceMetric(
    ,
                operation_id=f"export_{i}",
                response_time_ms=1000.0 + i * 100,
                success=True
            )
            monitor.record_performance_metric(metric)
        
        # Run a benchmark
        def simple_op():
    "
        
        monitor.run_benchmark("export3)
        
        # Export data
        export_data = monitor.export_performance_data(hours=1.0)
        
        assert export_data['tot== 5
        assert len(export_data['metrics']) == 5
        assert len(export_data['benchmarks']) == 1
        ta
        assert 'operation_cta
    
    @patch('intelligent_web_scraper.monitoring.performance_monitorcess')
    def monitor):
        """Test getting current """
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value. MB
    ce
        
        memory_usage = monitor._get_current_memory_usage()
        assert memory_12.0
    
    @patch('intelligent_web_scraper.moness')
    def test_get_current_cpu_usage(self, m:
        """Test getting current CPU "
        m)
        mock_process_instance.cpu_percent.return_value = 75.5
        
        
        cpu_usage = monitor._get_current_cpu_usage()
        
    
    @patch('intelligent_weess')
    def test_memory_cpu_usage_error_hanonitor):
        """Test error handling for memory and CP"
        # Test NoSuchProcess exception
        mock_process.side_efs(123)
        
        memory_usage = monitor._get_current_memory_usge()
        
        
        assert memory_usage == 0.0
        assert cpu_usage == 0.0
    
    onitor):
        """Test performance threshold checking and"
        callback_events = []
        
        
            callback_events.append(eve
        
        callback)
        
        holds
        critical_metric = PerformanceMetric(
            operation_type="thresho,
            operation_id="critical_001",
            response_time_ms=15000.0reshold
            memory_usagethreshold
         shold
            success=True
        )
        monitor.record_performance_metritric)
        
        # Manually trigger threshold checking
    ()
        
        # Check for alerts
        alert_events = [e foert']
        assert len(alert_events) > 0
        
        alert_event = alert_events[0]
        assert 'Critical response time' in 
        assert 'Critical memory usage' in ' '.joinerts'])
        assert 'Critical CPU'])
    
    def test_compare_benchmarks(self, monitor):
        rks."""
        # Create first bek
        def operation1():
            time.sleep(0.001)
            return "result1"
        
        brk(
            "benchmark_1", operation1, num_operatio
        )
        
        # Create second benchmark (simulate better performance)
        2():
            time.sleep(0.0005)  # Faster operati
            return "result2"
        
        benchmark2 = monitor.run_benchmark(
            "benchmark_2", operation2, num_operatiations=0
        )
        
    
        comparison = monitor.compare_benchmarks("benchmark_1", "benchmark_2")
        
        assert 'benchmark1' in comparison
        assert 'benchmark2' in comparison
        assert 'comparison' in comparin
        assert 'response_time_difference_ms' in comparison['comparison']
        assert 'throughput_difference_ops_per_sec' in comparion']
        assert 'overall_improvement' in comparison['compa
    
    def test_compare_benchmarks_not_found(self, monitor):
        """Test comparing benchmarks when one doesn'"""
        
        
        assert 'error' in comparison
    ison
    
    def test_get_resource_utilization_trends(self, monitor):
        """Test getting resource utilization trends."""
        # Add metrics over time
        base_time = datetime.utcnow() - timedelta(hours=1)
        
        for i in range(20):
            metric = PerformanceMetric(
        ),
                operation_type="trest",
                operation_id=f"}",
    * 50,
                memory_usage_mb=200.0 + i * 10,
                cpu_usage_percent=30.0 + i * 2,
                success=True
        
            monitor.record_performance_)
        
        0)
        
        trends
        assert 'cpu_trend' in trends
        assert 'response_time_trend' in trends
        assert 'interval_minutes' in trends
        assert 'total_data_points' in trends
        
        # Check that trends have data points
        assert len(trends['memory_trend']) > 0
        assert len(trend 0
        aend']) > 0
        
        # Check trend data structure
        memory_point = trends['memory_trend'][0]
        _point
        assert 'average' in memory_point
        assert 'min' in memory_point
        assert 'max' in memory_point
        ntemory_poiunt' in mssert 'coaemoryestamp' in m 'timertasse_time_trnds['respons len(tresserttrend']) >s['cpu_trend' in y_memor 'rtasses(hours=2._trendationlizource_utir.get_resitonds = montre(metricmetric )   00.0 + i _time_ms=10sponse       re     rend_{itend_t * 3tes=ielta(minuimed+ te base_tim timestamp=       n comparrks' inchmaavailable_be assert '   tent2")exisnontent1", "("nonexisrks_benchmaompareor.conitrison = mcompat exist.rison']is'comparson[sohmarksnce beparom    # Cermup_opar5, wons=onoperationdef rations=0 warmup_opens=5,benchmamonitor.run_chmark1 = enmarnchnchmaring two beompaest c"""Terts_event['aloin(alert in ' '.j usage'rt_event['al(ales'])['alertventoin(alert_e' '.jance_al= 'performype') =if e.get('tnts callback_ever e in holdshresrmance_t_check_perfo  monitor.  al_meicc(crititical thre crceeds   # Ext=98.0,enusage_perc cpu_  l ds critica Excee=1200.0,   #_mbal thicrit # Exceeds c, ld_test" thresedsic that exce metr Add#k(alert__callbacncermaerfoadd_por.monitta)nt_dat_data):ven(ert_callbackdef ale"" alerts.(self, mlds_checkingthreshorformance_f test_pede_usage()nt_cpurecur_get_onitor.sage = mcpu_uahProces.NoSuc= psutilct fe"."U usageess, m mock_procng(self,dliProcl.r.psutimance_monitorfor.peitoringscraper.monb_ == 75.5_usagert cpuassess_instanceoceprue = mock_aleturn_vs.resrocmock_p= Mock(nstance rocess_i_pockusage.""itor)ss, monock_procetil.Procor.psuce_monitan.performringitoage == 5usnstan_process_ick = moalueess.return_v   mock_proc 24  # 51224 * 10s = 512 * 10rsry usage.memocess, , mock_proy_usage(selfnt_memorget_curretest_ropsutil.P.port_daters' in exounport_dain exds' resholformance_thsert 'peras_metrics'] alerations=_op, num_oprk", simple_benchmaeturn "test        rport_test"pe="exion_ty      operat      :)back'] == 'callon_typeatioper']['s[0]['metric_eventlbackt calasser == vents)k_elbac(calssert lenametric(meance_s=True   succes00.0e_ms=10timesponse_ck_testllbak_cack(testance_callba_performaddr.allbacks.ent cr):self, monito_callbacks(ormanceerfeme 0  # Improv"] >_changeparisonmovemeImpr > 0  # ent"]provemmomparison in cmprovement"me_iiison()ce_compart_performangeonitor._= miccess        su     baselinelinehan base}","rationpe_ope="test_ty operation       _meneselibanes."""lise):orlf, monitines(sece_baselormanest_perfttable") == "sd([5.0]late_trenr._calcunitosert moase cases"stable == ues), 5.0, 5.0, 5.1, 4.9, 5.0 5.2, 4.8ecreasing""dues) == creasing_valrend(deculate_t 2.0, 5.0, 4.0, 3 7.0, 6.0, 8.0,.0, 9.0,in"increases) == g_valusincreate_trend(inr._calculasert monito    as 9.0, 10.0], 8.0,ulation."celf, moniate_trend(scult_caltestime_ms nse_ <= 100.0_scoreceanormperfall_overort.assert rep=1.0ric)        )g CPU u Increasin #y utimeresponse reasing nc0,  # I20ion_test"="optimizatypen_tperatio    o    manceerforic = P   metrticscharacterismance erforation reptimizr):tof, moni= '] =typeseration_['opmmaryassert su] == perations'l_ono data."th y wi summarmonitor):y(self, y_empt 80] ==_rate'esssucc]['t'summary_tes]['es'on_typatimary['operassert sum==] ]['count'est'ry_tsummaes']['typ'operation_[]== ns'] =_operatio['successfuls']_statry['overallummasert s   as (metetric_mformance_perrdor.reconitmo   sful, 2 faiucces20.0 + i *ent=* 1summaryion_type="      operat  cs0 50) == 5.le([5.0],idge cases# E, 99) = 9.55) == 95lues, 90) , == 5.5  #alues, 50)9.0, 1.0, 8.0,  5.0, 6.0, 70, 4.0,0, 3.2..0, es = [1luva."ionlculatentile ca"Test perc"lf, monle(se0.0ent < 10te_percess_ratns = =ations == 9erop.total_hmarksert bencasenchmark"ing_b"failark_name == nchmioatm_oper      nu  k"hmarnc "succurn    retall_cou  nonlocal c  "ons." operatingilis(self, month_failurek_wimarmarkenchnt_bcurre"con== me na.benchmark_arkenchmrt b    assebenchma"concurrent_rk_name=   benchma markeration()_opef testdme_ms > 0e_tinsns == 5eratiototal_opark.rt benchm    assemup_oper    warbenchmark",e="test_enchmark_nam    bsuccess"return "   )  # Simulat001.."chmark ben, monitor)l(self= "Test err =etricsormance_m.perf len(monitorert  ass   "Test errse,ck") as traor"op_err ",_operation"erroration(peritor.track_owith monocess_instaue = mock_prl45.0rn_value = etupu_percent.r_instance.cck_processmo1024 *  256 * lue.rss =."""ror handlingth er, monitorck_procesself, moror(sth_eresoc.Prr.psutilmonitormance_0time_ms > ic.response_metrassert     eration"_opimulate  # S1).sleep(0.0me       ti acker:tr001") as on", "op_test_operati"n(atioper_orackth monitor.tistancecess_inpro = mock_aluet.return_vacenper 1024  ** 1024256 rss = e = Mock()stancnsss metriceer."" managontext_operation ct trackes """T   , monitor):ss_procef, mockselProcess'tor.psutil._moninceg.performar.monitorincrape==p"] ers["test_oation_countr.operitoassert mon        "]) == 1"test_op(metrc,23""test_1=tion_id   opera     r):lf, monitotric(se_me is not Nring_threadmonito monitor.assertng.""monitoristopping g and rtin""Test sta   " f, monitl Falses= = 0s) =ance_metricor.performnit(moert lenssa""."onitor):n(self, mor_creatiof test_monitdeking=Trutailed_trace_de enably_size=100,or(historitonsting. temonitor forperformance "Create a      ""   r(senito mo    deffixtu   @pytest.


class TestOperationTracker:
    """Test OperationTracker helper class."""
    
    def test_operation_tracker_creation(self):
        """Test creating an operation tracker."""
        tracker = OperationTracker("op_123", "test_operation")
        
        assert tracker.operation_id == "op_123"
        assert tracker.operation_type == "test_operation"
        assert tracker.success is True
        assert tracker.error_message is None
        assert tracker.metadata == {}
    
    def test_set_success(self):
        """Test setting operation success status."""
        tracker = OperationTracker("op_456", "scraping")
        
        # Test successful operation
        tracker.set_success(True)
        assert tracker.success is True
        assert tracker.error_message is None
        
        # Test failed operation
        tracker.set_success(False, "Network error")
        assert tracker.success is False
        assert tracker.error_message == "Network error"
    
    def test_add_metadata(self):
        """Test adding metadata to operation."""
        tracker = OperationTracker("op_789", "analysis")
        
        tracker.add_metadata("url", "https://example.com")
        tracker.add_metadata("pages", 5)
        
        assert tracker.metadata["url"] == "https://example.com"
        assert tracker.metadata["pages"] == 5


class TestPerformanceMonitor:
    """Test PerformanceMonitor main class."""
    
    @pytest.fixture
    def monitor(self):
        """Create a performance monitor for testing."""
        return PerformanceMonitor(
            history_size=100,
            benchmark_retention_days=7,
            enable_detailed_tracking=True
        )
    
    def test_monitor_initialization(self, monitor):
        """Test performance monitor initialization."""
        assert monitor.history_size == 100
        assert monitor.benchmark_retention_days == 7
        assert monitor.enable_detailed_tracking is True
        assert monitor.is_monitoring is False
        assert len(monitor.performance_metrics) == 0
        assert len(monitor.benchmark_results) == 0
    
    def test_start_stop_monitoring(self, monitor):
        """Test starting and stopping monitoring."""
        # Test start monitoring
        monitor.start_monitoring()
        assert monitor.is_monitoring is True
        assert monitor.monitoring_thread is not None
        assert monitor.monitoring_thread.is_alive()
        
        # Test stop monitoring
        monitor.stop_monitoring()
        assert monitor.is_monitoring is False
        
        # Wait for thread to finish
        if monitor.monitoring_thread:
            monitor.monitoring_thread.join(timeout=2.0)
    
    def test_record_performance_metric(self, monitor):
        """Test recording a performance metric."""
        metric = PerformanceMetric(
            operation_type="test",
            operation_id="test_001",
            response_time_ms=1000.0,
            memory_usage_mb=128.0,
            success=True
        )
        
        monitor.record_performance_metric(metric)
        
        assert len(monitor.performance_metrics) == 1
        assert len(monitor.operation_metrics["test"]) == 1
        assert monitor.operation_counters["test"] == 1
        
        recorded_metric = monitor.performance_metrics[0]
        assert recorded_metric.operation_type == "test"
        assert recorded_metric.response_time_ms == 1000.0
    
    @patch('intelligent_web_scraper.monitoring.performance_monitor.psutil.Process')
    def test_track_operation_context_manager(self, mock_process, monitor):
        """Test the track_operation context manager."""
        # Mock process for memory and CPU usage
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 128 * 1024 * 1024  # 128MB
        mock_process_instance.cpu_percent.return_value = 25.0
        mock_process.return_value = mock_process_instance
        
        # Test successful operation
        with monitor.track_operation("test_op", "op_001") as tracker:
            time.sleep(0.01)  # Simulate some work
            tracker.set_success(True)
        
        assert len(monitor.performance_metrics) == 1
        metric = monitor.performance_metrics[0]
        assert metric.operation_type == "test_op"
        assert metric.operation_id == "op_001"
        assert metric.success is True
        assert metric.response_time_ms > 0
    
    @patch('intelligent_web_scraper.monitoring.performance_monitor.psutil.Process')
    def test_track_operation_with_error(self, mock_process, monitor):
        """Test tracking an operation that fails."""
        # Mock process
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 64 * 1024 * 1024
        mock_process_instance.cpu_percent.return_value = 15.0
        mock_process.return_value = mock_process_instance
        
        # Test failed operation
        with monitor.track_operation("failing_op", "fail_001") as tracker:
            tracker.set_success(False, "Test error")
        
        assert len(monitor.performance_metrics) == 1
        metric = monitor.performance_metrics[0]
        assert metric.success is False
        assert metric.error_message == "Test error"
    
    def test_run_benchmark_sequential(self, monitor):
        """Test running a sequential benchmark."""
        def test_operation():
            time.sleep(0.001)  # Simulate work
            return "result"
        
        benchmark = monitor.run_benchmark(
            benchmark_name="test_benchmark",
            operation_func=test_operation,
            num_operations=5,
            concurrent_operations=1,
            warmup_operations=2
        )
        
        assert benchmark.benchmark_name == "test_benchmark"
        assert benchmark.total_operations == 5
        assert benchmark.successful_operations == 5
        assert benchmark.failed_operations == 0
        assert benchmark.throughput_ops_per_sec > 0
        assert benchmark.average_response_time_ms > 0
        
        # Check that benchmark was stored
        assert len(monitor.benchmark_results) == 1
    
    def test_run_benchmark_concurrent(self, monitor):
        """Test running a concurrent benchmark."""
        def test_operation():
            time.sleep(0.001)
            return "concurrent_result"
        
        benchmark = monitor.run_benchmark(
            benchmark_name="concurrent_test",
            operation_func=test_operation,
            num_operations=10,
            concurrent_operations=3,
            warmup_operations=0
        )
        
        assert benchmark.benchmark_name == "concurrent_test"
        assert benchmark.total_operations == 10
        assert benchmark.throughput_ops_per_sec > 0
    
    def test_run_benchmark_with_failures(self, monitor):
        """Test benchmark with some failing operations."""
        call_count = 0
        
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Every 3rd operation fails
                raise Exception("Simulated failure")
            return "success"
        
        benchmark = monitor.run_benchmark(
            benchmark_name="failing_test",
            operation_func=failing_operation,
            num_operations=9,
            concurrent_operations=1,
            warmup_operations=0
        )
        
        assert benchmark.benchmark_name == "failing_test"
        assert benchmark.total_operations == 9
        assert benchmark.failed_operations == 3  # Every 3rd operation
        assert benchmark.successful_operations == 6
        assert benchmark.success_rate_percent == (6/9) * 100
    
    def test_calculate_percentile(self, monitor):
        """Test percentile calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        p50 = monitor._calculate_percentile(values, 50)
        p95 = monitor._calculate_percentile(values, 95)
        p99 = monitor._calculate_percentile(values, 99)
        
        assert p50 == 5.5  # Median
        assert abs(p95 - 9.55) < 0.01  # 95th percentile
        assert abs(p99 - 9.91) < 0.01  # 99th percentile
        
        # Test edge cases
        assert monitor._calculate_percentile([], 50) == 0.0
        assert monitor._calculate_percentile([5.0], 50) == 5.0
    
    def test_performance_summary(self, monitor):
        """Test getting performance summary."""
        # Add some test metrics
        for i in range(10):
            metric = PerformanceMetric(
                operation_type="summary_test",
                operation_id=f"op_{i}",
                response_time_ms=1000.0 + i * 100,
                success=i < 8  # 8 successful, 2 failed
            )
            monitor.record_performance_metric(metric)
        
        summary = monitor.get_performance_summary(hours=1.0)
        
        assert summary['total_operations'] == 10
        assert summary['successful_operations'] == 8
        assert summary['failed_operations'] == 2
        assert summary['success_rate_percent'] == 80.0
        assert summary['average_response_time_ms'] > 1000.0
        assert summary['min_response_time_ms'] == 1000.0
        assert summary['max_response_time_ms'] == 1900.0
        assert 'summary_test' in summary['operation_types']
    
    def test_performance_summary_empty(self, monitor):
        """Test performance summary with no data."""
        summary = monitor.get_performance_summary(hours=1.0)
        
        assert summary['total_operations'] == 0
        assert summary['average_response_time_ms'] == 0.0
        assert summary['throughput_ops_per_sec'] == 0.0
        assert summary['success_rate_percent'] == 0.0
    
    def test_generate_optimization_report(self, monitor):
        """Test generating optimization report."""
        # Add metrics with various performance characteristics
        metrics = [
            PerformanceMetric(
                operation_type="slow_op",
                response_time_ms=8000.0,  # Slow response
                memory_usage_mb=600.0,    # High memory
                cpu_usage_percent=85.0,   # High CPU
                success=True
            ),
            PerformanceMetric(
                operation_type="fast_op",
                response_time_ms=500.0,   # Fast response
                memory_usage_mb=100.0,    # Low memory
                cpu_usage_percent=20.0,   # Low CPU
                success=True
            ),
            PerformanceMetric(
                operation_type="failing_op",
                response_time_ms=2000.0,
                success=False,
                error_message="Test failure"
            )
        ]
        
        for metric in metrics:
            monitor.record_performance_metric(metric)
        
        report = monitor.generate_optimization_report(analysis_hours=1.0)
        
        assert isinstance(report, PerformanceOptimizationReport)
        assert report.analysis_period_hours == 1.0
        assert report.overall_performance_score >= 0.0
        assert report.overall_performance_score <= 100.0
        assert report.performance_trend in ["improving", "stable", "degrading", "mixed", "unknown"]
        assert report.average_response_time_ms > 0
    
    def test_generate_optimization_report_empty(self, monitor):
        """Test optimization report with no data."""
        report = monitor.generate_optimization_report(analysis_hours=1.0)
        
        assert report.overall_performance_score == 0.0
        assert report.performance_trend == "unknown"
        assert report.average_response_time_ms == 0.0
        assert len(report.optimization_recommendations) == 0
    
    def test_calculate_trend(self, monitor):
        """Test trend calculation."""
        # Test increasing trend
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        assert monitor._calculate_trend(increasing_values) == "increasing"
        
        # Test decreasing trend
        decreasing_values = [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        assert monitor._calculate_trend(decreasing_values) == "decreasing"
        
        # Test stable trend
        stable_values = [5.0, 5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9, 5.0, 5.0]
        assert monitor._calculate_trend(stable_values) == "stable"
        
        # Test insufficient data
        assert monitor._calculate_trend([1.0, 2.0]) == "stable"
        assert monitor._calculate_trend([]) == "stable"
    
    def test_performance_baselines(self, monitor):
        """Test setting and using performance baselines."""
        # Set baseline
        baseline_metrics = {
            'response_time_ms': 2000.0,
            'throughput_ops_per_sec': 5.0,
            'memory_usage_mb': 200.0
        }
        monitor.set_performance_baseline("test_operation", baseline_metrics)
        
        assert "test_operation" in monitor.performance_baselines
        assert monitor.performance_baselines["test_operation"]["response_time_ms"] == 2000.0
        
        # Add some current metrics
        for i in range(5):
            metric = PerformanceMetric(
                operation_type="test_operation",
                response_time_ms=1500.0,  # Better than baseline
                success=True
            )
            monitor.record_performance_metric(metric)
        
        # Get comparison
        comparison = monitor._get_performance_comparison()
        assert "test_operation_response_time_improvement" in comparison
        assert comparison["test_operation_response_time_improvement"] > 0  # Improvement
    
    def test_performance_callbacks(self, monitor):
        """Test performance event callbacks."""
        callback_events = []
        
        def test_callback(event_data):
            callback_events.append(event_data)
        
        monitor.add_performance_callback(test_callback)
        
        # Record a metric to trigger callback
        metric = PerformanceMetric(
            operation_type="callback_test",
            response_time_ms=1000.0,
            success=True
        )
        monitor.record_performance_metric(metric)
        
        assert len(callback_events) == 1
        assert callback_events[0]['type'] == 'metric_recorded'
        assert callback_events[0]['metric']['operation_type'] == 'callback_test'
    
    def test_export_performance_data(self, monitor):
        """Test exporting performance data."""
        # Add some test data
        for i in range(5):
            metric = PerformanceMetric(
                operation_type="export_test",
                operation_id=f"export_{i}",
                response_time_ms=1000.0 + i * 100,
                success=True
            )
            monitor.record_performance_metric(metric)
        
        # Add a benchmark
        benchmark = PerformanceBenchmark(
            benchmark_name="export_benchmark",
            total_operations=10,
            throughput_ops_per_sec=8.5
        )
        monitor.benchmark_results.append(benchmark)
        
        # Export data
        export_data = monitor.export_performance_data(hours=1.0)
        
        assert 'export_timestamp' in export_data
        assert export_data['period_hours'] == 1.0
        assert export_data['total_metrics'] == 5
        assert len(export_data['metrics']) == 5
        assert len(export_data['benchmarks']) == 1
        assert 'performance_thresholds' in export_data
        assert export_data['operation_counters']['export_test'] == 5
    
    @patch('intelligent_web_scraper.monitoring.performance_monitor.psutil.Process')
    def test_memory_and_cpu_tracking(self, mock_process, monitor):
        """Test memory and CPU usage tracking."""
        # Mock process with specific values
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 512 * 1024 * 1024  # 512MB
        mock_process_instance.cpu_percent.return_value = 75.0
        mock_process.return_value = mock_process_instance
        
        memory_usage = monitor._get_current_memory_usage()
        cpu_usage = monitor._get_current_cpu_usage()
        
        assert memory_usage == 512.0  # MB
        assert cpu_usage == 75.0      # Percent
    
    @patch('intelligent_web_scraper.monitoring.performance_monitor.psutil.Process')
    def test_memory_cpu_tracking_error_handling(self, mock_process, monitor):
        """Test error handling in memory/CPU tracking."""
        # Mock process to raise exception
        mock_process.side_effect = Exception("Process not found")
        
        memory_usage = monitor._get_current_memory_usage()
        cpu_usage = monitor._get_current_cpu_usage()
        
        assert memory_usage == 0.0
        assert cpu_usage == 0.0
    
    def test_performance_thresholds_checking(self, monitor):
        """Test performance threshold checking and alerts."""
        callback_events = []
        
        def alert_callback(event_data):
            callback_events.append(event_data)
        
        monitor.add_performance_callback(alert_callback)
        
        # Create metric that exceeds thresholds
        critical_metric = PerformanceMetric(
            operation_type="critical_test",
            operation_id="critical_001",
            response_time_ms=15000.0,  # Exceeds critical threshold
            memory_usage_mb=1200.0,    # Exceeds critical threshold
            cpu_usage_percent=98.0,    # Exceeds critical threshold
            success=True
        )
        
        monitor.record_performance_metric(critical_metric)
        monitor._check_performance_thresholds()
        
        # Should have received alert callbacks
        alert_events = [e for e in callback_events if e.get('type') == 'performance_alert']
        assert len(alert_events) > 0
        
        alert_event = alert_events[0]
        assert alert_event['operation_id'] == 'critical_001'
        assert len(alert_event['alerts']) > 0
        assert any('Critical response time' in alert for alert in alert_event['alerts'])
    
    def test_cleanup_old_data(self, monitor):
        """Test cleanup of old performance data."""
        # Add old benchmark
        old_benchmark = PerformanceBenchmark(
            benchmark_name="old_benchmark",
            timestamp=datetime.utcnow() - timedelta(days=40)  # Older than retention
        )
        monitor.benchmark_results.append(old_benchmark)
        
        # Add recent benchmark
        recent_benchmark = PerformanceBenchmark(
            benchmark_name="recent_benchmark",
            timestamp=datetime.utcnow() - timedelta(days=5)  # Within retention
        )
        monitor.benchmark_results.append(recent_benchmark)
        
        assert len(monitor.benchmark_results) == 2
        
        # Run cleanup
        monitor._cleanup_old_data()
        
        # Should only have recent benchmark
        assert len(monitor.benchmark_results) == 1
        assert monitor.benchmark_results[0].benchmark_name == "recent_benchmark"


class TestPerformanceOptimizationReport:
    """Test PerformanceOptimizationReport model."""
    
    def test_optimization_report_creation(self):
        """Test creating an optimization report."""
        report = PerformanceOptimizationReport(
            report_id="test_report_123",
            generated_at=datetime.utcnow(),
            analysis_period_hours=24.0,
            overall_performance_score=75.5,
            performance_trend="improving",
            average_response_time_ms=1200.0,
            response_time_trend="decreasing",
            throughput_ops_per_sec=8.5,
            throughput_trend="increasing",
            resource_utilization_percent=45.0,
            identified_bottlenecks=["High memory usage"],
            optimization_recommendations=["Implement caching", "Optimize queries"]
        )
        
        assert report.report_id == "test_report_123"
        assert report.overall_performance_score == 75.5
        assert report.performance_trend == "improving"
        assert len(report.identified_bottlenecks) == 1
        assert len(report.optimization_recommendations) == 2
    
    def test_optimization_report_defaults(self):
        """Test optimization report with default values."""
        report = PerformanceOptimizationReport(
            report_id="default_test",
            generated_at=datetime.utcnow(),
            analysis_period_hours=1.0,
            overall_performance_score=50.0,
            performance_trend="stable",
            average_response_time_ms=2000.0,
            response_time_trend="stable",
            throughput_ops_per_sec=2.0,
            throughput_trend="stable",
            resource_utilization_percent=30.0
        )
        
        assert len(report.identified_bottlenecks) == 0
        assert len(report.performance_issues) == 0
        assert len(report.optimization_recommendations) == 0
        assert len(report.configuration_suggestions) == 0
        assert len(report.performance_comparison) == 0
        assert len(report.benchmark_results) == 0


if __name__ == "__main__":
    pytest.main([__file__])       
 )
        
        assert benchmark.benchmark_name == "concurrent_test"
        assert benchmark.total_operations == 10
        assert benchmark.throughput_ops_per_sec > 0
    
    def test_run_benchmark_with_failures(self, monitor):
        """Test benchmark with some operations failing."""
        call_count = 0
        
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Every 3rd operation fails
                raise Exception("Simulated failure")
            return "success"
        
        benchmark = monitor.run_benchmark(
            benchmark_name="failing_benchmark",
            operation_func=failing_operation,
            num_operations=9,
            concurrent_operations=1,
            warmup_operations=0
        )
        
        assert benchmark.benchmark_name == "failing_benchmark"
        assert benchmark.total_operations == 9
        assert benchmark.failed_operations == 3
        assert benchmark.successful_operations == 6
        assert benchmark.error_rate_percent > 0
        assert benchmark.success_rate_percent < 100.0
    
    def test_get_performance_summary(self, monitor):
        """Test getting performance summary."""
        # Add test metrics
        for i in range(10):
            metric = PerformanceMetric(
                operation_type="summary_test",
                operation_id=f"sum_{i}",
                response_time_ms=1000.0 + i * 100,
                memory_usage_mb=100.0 + i * 10,
                cpu_usage_percent=20.0 + i * 5,
                success=i < 8  # 8 successful, 2 failed
            )
            monitor.record_performance_metric(metric)
        
        summary = monitor.get_performance_summary(hours=1.0)
        
        assert summary['total_operations'] == 10
        assert summary['overall_stats']['successful_operations'] == 8
        assert summary['overall_stats']['success_rate'] == 80.0
        assert 'summary_test' in summary['operation_types']
        assert summary['operation_types']['summary_test']['count'] == 10
        assert summary['operation_types']['summary_test']['success_rate'] == 80.0
    
    def test_set_performance_baseline(self, monitor):
        """Test setting and using performance baselines."""
        baseline_metrics = {
            'response_time_ms': 2000.0,
            'throughput_ops_per_sec': 5.0,
            'memory_usage_mb': 200.0
        }
        
        monitor.set_performance_baseline("baseline_test", baseline_metrics)
        
        # Add current metrics (better performance)
        for i in range(5):
            metric = PerformanceMetric(
                operation_type="baseline_test",
                operation_id=f"baseline_{i}",
                response_time_ms=1500.0,  # Better than baseline
                throughput_ops_per_sec=7.0,  # Better than baseline
                memory_usage_mb=150.0,  # Better than baseline
                success=True
            )
            monitor.record_performance_metric(metric)
        
        comparison = monitor._get_performance_comparison()
        
        assert "baseline_test_response_time_improvement" in comparison
        assert comparison["baseline_test_response_time_improvement"] > 0
        assert "baseline_test_throughput_change" in comparison
        assert comparison["baseline_test_throughput_change"] > 0
    
    def test_compare_benchmarks(self, monitor):
        """Test comparing two benchmarks."""
        # Create first benchmark
        def operation1():
            time.sleep(0.002)
            return "result1"
        
        benchmark1 = monitor.run_benchmark(
            "benchmark_1", operation1, num_operations=5, warmup_operations=0
        )
        
        # Create second benchmark (faster)
        def operation2():
            time.sleep(0.001)
            return "result2"
        
        benchmark2 = monitor.run_benchmark(
            "benchmark_2", operation2, num_operations=5, warmup_operations=0
        )
        
        comparison = monitor.compare_benchmarks("benchmark_1", "benchmark_2")
        
        assert 'benchmark1' in comparison
        assert 'benchmark2' in comparison
        assert 'comparison' in comparison
        assert 'response_time_difference_ms' in comparison['comparison']
        assert 'throughput_difference_ops_per_sec' in comparison['comparison']
        assert 'overall_improvement' in comparison['comparison']
    
    def test_compare_benchmarks_not_found(self, monitor):
        """Test comparing non-existent benchmarks."""
        comparison = monitor.compare_benchmarks("nonexistent1", "nonexistent2")
        
        assert 'error' in comparison
        assert 'available_benchmarks' in comparison
    
    def test_get_resource_utilization_trends(self, monitor):
        """Test getting resource utilization trends."""
        # Add metrics over time
        base_time = datetime.utcnow() - timedelta(hours=1)
        
        for i in range(20):
            metric = PerformanceMetric(
                timestamp=base_time + timedelta(minutes=i * 3),
                operation_type="trend_test",
                operation_id=f"trend_{i}",
                response_time_ms=1000.0 + i * 50,
                memory_usage_mb=200.0 + i * 10,
                cpu_usage_percent=30.0 + i * 2,
                success=True
            )
            monitor.record_performance_metric(metric)
        
        trends = monitor.get_resource_utilization_trends(hours=2.0)
        
        assert 'memory_trend' in trends
        assert 'cpu_trend' in trends
        assert 'response_time_trend' in trends
        assert 'interval_minutes' in trends
        assert 'total_data_points' in trends
        
        # Check that trends have data points
        assert len(trends['memory_trend']) > 0
        assert len(trends['cpu_trend']) > 0
        assert len(trends['response_time_trend']) > 0
        
        # Check trend data structure
        if trends['memory_trend']:
            memory_point = trends['memory_trend'][0]
            assert 'timestamp' in memory_point
            assert 'average' in memory_point
            assert 'min' in memory_point
            assert 'max' in memory_point
            assert 'count' in memory_point
    
    def test_generate_optimization_report(self, monitor):
        """Test generating optimization report."""
        # Add metrics with performance issues
        for i in range(15):
            metric = PerformanceMetric(
                operation_type="optimization_test",
                operation_id=f"opt_{i}",
                response_time_ms=3000.0 + i * 200,  # High response times
                memory_usage_mb=400.0 + i * 30,     # High memory usage
                cpu_usage_percent=60.0 + i * 3,     # High CPU usage
                success=i < 12  # Some failures
            )
            monitor.record_performance_metric(metric)
        
        report = monitor.generate_optimization_report(analysis_hours=1.0)
        
        assert report.report_id.startswith("opt_report_")
        assert report.analysis_period_hours == 1.0
        assert report.overall_performance_score >= 0.0
        assert report.overall_performance_score <= 100.0
        assert report.average_response_time_ms > 0
        assert len(report.optimization_recommendations) > 0
    
    def test_performance_callbacks(self, monitor):
        """Test performance event callbacks."""
        callback_events = []
        
        def test_callback(event_data):
            callback_events.append(event_data)
        
        monitor.add_performance_callback(test_callback)
        
        # Record a metric to trigger callback
        metric = PerformanceMetric(
            operation_type="callback_test",
            operation_id="callback_001",
            response_time_ms=1000.0,
            success=True
        )
        monitor.record_performance_metric(metric)
        
        assert len(callback_events) == 1
        assert callback_events[0]['type'] == 'metric_recorded'
        assert callback_events[0]['metric']['operation_type'] == 'callback_test'
    
    def test_export_performance_data(self, monitor):
        """Test exporting performance data."""
        # Add test data
        for i in range(5):
            metric = PerformanceMetric(
                operation_type="export_test",
                operation_id=f"export_{i}",
                response_time_ms=1000.0 + i * 100,
                success=True
            )
            monitor.record_performance_metric(metric)
        
        # Run a benchmark
        def simple_op():
            return "test"
        
        monitor.run_benchmark("export_benchmark", simple_op, num_operations=3)
        
        # Export data
        export_data = monitor.export_performance_data(hours=1.0)
        
        assert export_data['total_metrics'] == 5
        assert len(export_data['metrics']) == 5
        assert len(export_data['benchmarks']) == 1
        assert 'performance_thresholds' in export_data
        assert 'operation_counters' in export_data
    
    @patch('intelligent_web_scraper.monitoring.performance_monitor.psutil.Process')
    def test_memory_cpu_usage_methods(self, mock_process, monitor):
        """Test memory and CPU usage collection methods."""
        # Test successful collection
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 256 * 1024 * 1024  # 256MB
        mock_process_instance.cpu_percent.return_value = 45.0
        mock_process.return_value = mock_process_instance
        
        memory_usage = monitor._get_current_memory_usage()
        cpu_usage = monitor._get_current_cpu_usage()
        
        assert memory_usage == 256.0
        assert cpu_usage == 45.0
        
        # Test error handling
        mock_process.side_effect = psutil.NoSuchProcess(123)
        
        memory_usage = monitor._get_current_memory_usage()
        cpu_usage = monitor._get_current_cpu_usage()
        
        assert memory_usage == 0.0
        assert cpu_usage == 0.0
    
    def test_performance_threshold_alerts(self, monitor):
        """Test performance threshold checking and alerts."""
        callback_events = []
        
        def alert_callback(event_data):
            callback_events.append(event_data)
        
        monitor.add_performance_callback(alert_callback)
        
        # Add metric that exceeds thresholds
        critical_metric = PerformanceMetric(
            operation_type="threshold_test",
            operation_id="critical_001",
            response_time_ms=12000.0,  # Exceeds critical threshold
            memory_usage_mb=1100.0,   # Exceeds critical threshold
            cpu_usage_percent=97.0,   # Exceeds critical threshold
            success=True
        )
        monitor.record_performance_metric(critical_metric)
        
        # Manually trigger threshold checking
        monitor._check_performance_thresholds()
        
        # Check for alerts
        alert_events = [e for e in callback_events if e.get('type') == 'performance_alert']
        assert len(alert_events) > 0
        
        alert_event = alert_events[0]
        alerts_text = ' '.join(alert_event['alerts'])
        assert 'Critical response time' in alerts_text
        assert 'Critical memory usage' in alerts_text
        assert 'Critical CPU usage' in alerts_text
    
    def test_calculate_percentile_edge_cases(self, monitor):
        """Test percentile calculation with edge cases."""
        # Normal case
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert monitor._calculate_percentile(values, 50) == 3.0
        
        # Single value
        assert monitor._calculate_percentile([5.0], 50) == 5.0
        
        # Empty list
        assert monitor._calculate_percentile([], 50) == 0.0
        
        # Test various percentiles
        values = list(range(1, 101))  # 1 to 100
        assert monitor._calculate_percentile(values, 95) == 95.05
        assert monitor._calculate_percentile(values, 99) == 99.01
    
    def test_calculate_trend_variations(self, monitor):
        """Test trend calculation with various patterns."""
        # Strong increasing trend
        increasing = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]
        assert monitor._calculate_trend(increasing) == "increasing"
        
        # Strong decreasing trend
        decreasing = [20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0]
        assert monitor._calculate_trend(decreasing) == "decreasing"
        
        # Stable trend
        stable = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 10.0]
        assert monitor._calculate_trend(stable) == "stable"
        
        # Insufficient data
        assert monitor._calculate_trend([1.0, 2.0]) == "stable"
        assert monitor._calculate_trend([]) == "stable"
    
    def test_calculate_overall_improvement(self, monitor):
        """Test overall improvement calculation between benchmarks."""
        # Create two benchmarks for comparison
        benchmark1 = PerformanceBenchmark(
            benchmark_name="baseline",
            average_response_time_ms=2000.0,
            throughput_ops_per_sec=5.0,
            success_rate_percent=90.0,
            average_memory_mb=200.0
        )
        
        # Better benchmark
        benchmark2 = PerformanceBenchmark(
            benchmark_name="improved",
            average_response_time_ms=1500.0,  # Better (lower)
            throughput_ops_per_sec=7.0,      # Better (higher)
            success_rate_percent=95.0,       # Better (higher)
            average_memory_mb=150.0          # Better (lower)
        )
        
        improvement = monitor._calculate_overall_improvement(benchmark1, benchmark2)
        assert improvement == "significant_improvement"
        
        # Worse benchmark
        benchmark3 = PerformanceBenchmark(
            benchmark_name="degraded",
            average_response_time_ms=2500.0,  # Worse (higher)
            throughput_ops_per_sec=3.0,      # Worse (lower)
            success_rate_percent=85.0,       # Worse (lower)
            average_memory_mb=250.0          # Worse (higher)
        )
        
        improvement = monitor._calculate_overall_improvement(benchmark1, benchmark3)
        assert improvement == "degradation"
    
    def test_cleanup_old_data(self, monitor):
        """Test cleanup of old performance data."""
        # Set short retention period for testing
        monitor.benchmark_retention_days = 0.001  # Very short for testing
        
        # Add old benchmark
        old_benchmark = PerformanceBenchmark(
            benchmark_name="old_benchmark",
            timestamp=datetime.utcnow() - timedelta(days=1)
        )
        monitor.benchmark_results.append(old_benchmark)
        
        # Add recent benchmark
        recent_benchmark = PerformanceBenchmark(
            benchmark_name="recent_benchmark",
            timestamp=datetime.utcnow()
        )
        monitor.benchmark_results.append(recent_benchmark)
        
        assert len(monitor.benchmark_results) == 2
        
        # Trigger cleanup
        monitor._cleanup_old_data()
        
        # Only recent benchmark should remain
        assert len(monitor.benchmark_results) == 1
        assert monitor.benchmark_results[0].benchmark_name == "recent_benchmark"


class TestPerformanceOptimizationReport:
    """Test PerformanceOptimizationReport model."""
    
    def test_optimization_report_creation(self):
        """Test creating an optimization report."""
        report = PerformanceOptimizationReport(
            report_id="test_report_123",
            generated_at=datetime.utcnow(),
            analysis_period_hours=24.0,
            overall_performance_score=75.5,
            performance_trend="improving",
            average_response_time_ms=1500.0,
            response_time_trend="decreasing",
            throughput_ops_per_sec=8.2,
            throughput_trend="increasing",
            resource_utilization_percent=65.0,
            identified_bottlenecks=["High memory usage", "Slow database queries"],
            optimization_recommendations=["Implement caching", "Optimize queries"],
            configuration_suggestions={"max_connections": 100, "cache_size": "512MB"}
        )
        
        assert report.report_id == "test_report_123"
        assert report.overall_performance_score == 75.5
        assert report.performance_trend == "improving"
        assert len(report.identified_bottlenecks) == 2
        assert len(report.optimization_recommenstionuggesfiguration_sort.conin repections" t "max_conn asser 2
       dations) ==