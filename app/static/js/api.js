(function () {
  function withTimeout(promise, timeoutMs, message) {
    return new Promise((resolve, reject) => {
      const timer = window.setTimeout(() => reject(new Error(message || 'Request timed out.')), timeoutMs);
      promise
        .then((value) => {
          window.clearTimeout(timer);
          resolve(value);
        })
        .catch((error) => {
          window.clearTimeout(timer);
          reject(error);
        });
    });
  }

  window.AppApi = {
    withTimeout,
  };
})();
