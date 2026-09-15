/* Fills the download button from the newest release: the zip's address, the version, the size
   and the digest. Without a reply the page reads as written, with the Releases page as the
   button's address. Under a reduced-motion preference the recording waits to be played. */
(function () {
  "use strict";

  var ASSET = "twinscribe-win64.zip";
  var RELEASES = "https://api.github.com/repos/Unified-Source/TwinScribe/releases?per_page=1";

  var video = document.getElementById("run");
  if (video && window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    video.removeAttribute("autoplay");
    video.pause();
  }

  var button = document.getElementById("download");
  var detail = document.getElementById("download-detail");
  var digest = document.getElementById("download-digest");
  if (!button || !detail || !window.fetch) {
    return;
  }

  function gigabytes(bytes) {
    return (bytes / 1e9).toFixed(2) + " GB";
  }

  fetch(RELEASES, { headers: { Accept: "application/vnd.github+json" } })
    .then(function (reply) {
      return reply.ok ? reply.json() : Promise.reject(reply.status);
    })
    .then(function (releases) {
      var release = releases && releases[0];
      var asset = release && (release.assets || []).filter(function (candidate) {
        return candidate.name === ASSET;
      })[0];
      if (!asset) {
        return;
      }
      button.href = asset.browser_download_url;
      var text = ASSET + ", " + release.tag_name + ", " + gigabytes(asset.size);
      if (release.prerelease) {
        text += ", a pre-release";
      }
      detail.textContent = text + "; the models are fetched on first start.";
      if (digest && asset.digest) {
        digest.textContent = asset.digest.replace("sha256:", "sha256 ");
        digest.hidden = false;
      }
    })
    .catch(function () {
      /* The page already says where the zip is. */
    });
})();
