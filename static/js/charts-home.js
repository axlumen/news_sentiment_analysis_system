// charts-home.js
(function($) {
  'use strict';
  $(function() {
    // 情感分析趋势图表
    var lineChart1 = $('#lineChart1');
    if (lineChart1.length) {
      var ctx = lineChart1.get(0).getContext('2d');
      var chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: ['周一', '周二', '周三', '周四', '周五', '周六', '周日'],
          datasets: [
            {
              label: '正面情感',
              data: [sentimentData.positive * 0.8, sentimentData.positive * 0.9, sentimentData.positive, sentimentData.positive * 1.1, sentimentData.positive * 1.2, sentimentData.positive * 1.1, sentimentData.positive],
              backgroundColor: 'rgba(75, 192, 192, 0.2)',
              borderColor: 'rgba(75, 192, 192, 1)',
              borderWidth: 1
            },
            {
              label: '负面情感',
              data: [sentimentData.negative * 0.8, sentimentData.negative * 0.9, sentimentData.negative, sentimentData.negative * 1.1, sentimentData.negative * 0.9, sentimentData.negative * 0.8, sentimentData.negative],
              backgroundColor: 'rgba(255, 99, 132, 0.2)',
              borderColor: 'rgba(255, 99, 132, 1)',
              borderWidth: 1
            },
            {
              label: '中性情感',
              data: [sentimentData.neutral * 0.8, sentimentData.neutral * 0.9, sentimentData.neutral, sentimentData.neutral * 1.1, sentimentData.neutral * 0.9, sentimentData.neutral * 1.1, sentimentData.neutral],
              backgroundColor: 'rgba(54, 162, 235, 0.2)',
              borderColor: 'rgba(54, 162, 235, 1)',
              borderWidth: 1
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false
        }
      });
    }

    // 新闻来源分布图表
    var lineChart2 = $('#lineChart2');
    if (lineChart2.length) {
      var ctx = lineChart2.get(0).getContext('2d');
      var chart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: Object.keys(sourceData),
          datasets: [{
            label: '新闻数量',
            data: Object.values(sourceData),
            backgroundColor: [
              'rgba(255, 99, 132, 0.2)',
              'rgba(54, 162, 235, 0.2)',
              'rgba(255, 206, 86, 0.2)',
              'rgba(75, 192, 192, 0.2)',
              'rgba(153, 102, 255, 0.2)',
              'rgba(255, 159, 64, 0.2)'
            ],
            borderColor: [
              'rgba(255, 99, 132, 1)',
              'rgba(54, 162, 235, 1)',
              'rgba(255, 206, 86, 1)',
              'rgba(75, 192, 192, 1)',
              'rgba(153, 102, 255, 1)',
              'rgba(255, 159, 64, 1)'
            ],
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false
        }
      });
    }
  });
})(jQuery);
