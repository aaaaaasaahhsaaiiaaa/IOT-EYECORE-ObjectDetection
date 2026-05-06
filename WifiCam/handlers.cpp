#include "WifiCam.hpp"
#include <StreamString.h>
#include <uri/UriBraces.h>

static const char FRONTPAGE[] PROGMEM = R"EOT(
<!doctype html>
<title>esp32cam WifiCam example</title>
<style>
table,th,td { border: solid 1px #000; border-collapse: collapse; }
th,td { padding: 0.4rem; }
a { text-decoration: none; }
footer { margin-top: 1rem; }
</style>
<body>
<h1>esp32cam WifiCam example</h1>

<table>
<thead>
<tr><th>BMP<th>JPG<th>MJPEG</tr>
<tbody id="resolutions">
<tr><td colspan="3">loading</td></tr>
</tbody>
</table>

<footer>Powered by esp32cam</footer>

<script type="module">
async function fetchText(uri) {
  const res = await fetch(uri);
  if (!res.ok) throw new Error(await res.text());
  return (await res.text()).trim();
}

try {
  const list = (await fetchText("/resolutions.csv")).split("\n");
  document.querySelector("#resolutions").innerHTML =
    list.map(r => `<tr>${
      ["bmp","jpg","mjpeg"].map(fmt =>
        `<td><a href="/${r}.${fmt}">${r}</a></td>`
      ).join("")
    }</tr>`).join("");
} catch (e) {
  document.querySelector("#resolutions").innerHTML =
    `<tr><td colspan="3">${e}</td></tr>`;
}
</script>
)EOT";