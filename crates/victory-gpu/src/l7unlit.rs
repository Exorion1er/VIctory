//! Lot 7, unlit mode. Everything is off unless its variable is set:
//!
//! - `VICTORY_L7_ROLES=<file>`: shader roles by hash, one `hash role [name]`
//!   per line (`#` comments), as `anyxboxx/patches-l7-unlit/roles.tsv` has
//!   them (made from the names in `shaders.rpf`). The two below need it.
//! - `VICTORY_L7_SKIP=<role>,...`: draws and dispatches of a pipeline with a
//!   shader of one of those roles are skipped, as `VICTORY_SKIP_PIPELINES`
//!   skips by hash (builds with diagnostics).
//! - `VICTORY_L7_UNLIT=replace|add`: the first time a list draws with a
//!   lighting pipeline (role `lighting`: Grand Theft Auto VI's tiled
//!   lighting pixel shaders, `TL_PS_*`) into a target in a frame, a pass of
//!   ours first writes the albedo of the G-buffer (the targets of the last
//!   draws with a `gbuffer` pixel shader, of the same size) times
//!   `VICTORY_L7_UNLIT_GAIN` into that target's colour. `replace` then drops
//!   the lighting draws (no lights, no shadows: a flat, fully textured
//!   image); `add` keeps them and adds the albedo under them (an ambient
//!   floor for what the missing ray-traced light leaves black).
//!   `VICTORY_L7_UNLIT_ALBEDO=<n>` (0): which G-buffer target holds the
//!   albedo. `VICTORY_L7_UNLIT_DECODE=auto|linear|square|gamma` (auto:
//!   square for 8/10-bit UNORM targets, as stored with a square-root
//!   encoding; linear for the rest, sRGB views decode themselves).
//!   `VICTORY_L7_UNLIT_FAR=0|1|off` (0): where the lighting draws have a
//!   depth target, only texels whose depth is not this far value (the sky)
//!   are written; `off` writes every texel.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::object::This;

macro_rules! l7_once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// The role table: role names (at most 64), and each shader hash's role.
struct Roles {
    names: Vec<String>,
    by_hash: HashMap<String, u8>,
}

fn roles() -> &'static Roles {
    static ROLES: OnceLock<Roles> = OnceLock::new();
    ROLES.get_or_init(|| {
        let mut roles = Roles {
            names: Vec::new(),
            by_hash: HashMap::new(),
        };
        let Some(path) = std::env::var_os("VICTORY_L7_ROLES") else {
            return roles;
        };
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!(path = ?path, error = %e, "VICTORY_L7_ROLES: cannot read the role table; no roles");
                return roles;
            }
        };
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut words = line.split_whitespace();
            let (Some(hash), Some(role)) = (words.next(), words.next()) else {
                continue;
            };
            if hash.len() != 16 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                continue;
            }
            let index = match roles.names.iter().position(|n| n == role) {
                Some(i) => i,
                None if roles.names.len() < 64 => {
                    roles.names.push(role.to_owned());
                    roles.names.len() - 1
                }
                None => {
                    l7_once!(role, "VICTORY_L7_ROLES: more than 64 roles; the rest are left out");
                    continue;
                }
            };
            roles.by_hash.insert(hash.to_ascii_lowercase(), index as u8);
        }
        roles
    })
}

/// The bit of role `name`, 0 if the table has no such role.
fn bit(name: &str) -> u64 {
    roles()
        .names
        .iter()
        .position(|n| n == name)
        .map_or(0, |i| 1 << i)
}

/// The roles of the shaders `hashes` (a pipeline's), as bits.
pub(crate) fn mask<'a>(hashes: impl Iterator<Item = &'a str>) -> u64 {
    let r = roles();
    if r.by_hash.is_empty() {
        return 0;
    }
    hashes
        .filter_map(|h| r.by_hash.get(h))
        .fold(0, |m, &i| m | 1 << i)
}

/// What `VICTORY_L7_UNLIT` asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unlit {
    /// The lighting draws dropped, the albedo in their place.
    Replace,
    /// The lighting draws kept, the albedo added under them.
    Add,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decode {
    Auto,
    Linear,
    Square,
    Gamma,
}

struct Config {
    skip: u64,
    unlit: Option<Unlit>,
    lighting: u64,
    gbuffer: u64,
    gain: f32,
    albedo: usize,
    decode: Decode,
    far: Option<f32>,
}

fn config() -> &'static Config {
    static CONFIG: OnceLock<Config> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let var = |name: &str| std::env::var(name).ok().map(|v| v.trim().to_ascii_lowercase());
        let mut skip = 0;
        for role in var("VICTORY_L7_SKIP").unwrap_or_default().split(',').map(str::trim) {
            if role.is_empty() {
                continue;
            }
            let b = bit(role);
            if b == 0 {
                tracing::warn!(role, "VICTORY_L7_SKIP: no such role in the VICTORY_L7_ROLES table");
            }
            skip |= b;
        }
        let unlit = match var("VICTORY_L7_UNLIT").as_deref() {
            None | Some("" | "0" | "off") => None,
            Some("add") => Some(Unlit::Add),
            Some("1" | "replace" | "on") => Some(Unlit::Replace),
            Some(other) => {
                tracing::warn!(value = other, "VICTORY_L7_UNLIT: replace, add or 0; taken as replace");
                Some(Unlit::Replace)
            }
        };
        let (lighting, gbuffer) = (bit("lighting"), bit("gbuffer"));
        if unlit.is_some() && (lighting == 0 || gbuffer == 0) {
            tracing::warn!("VICTORY_L7_UNLIT needs the lighting and gbuffer roles of VICTORY_L7_ROLES: off");
        }
        let unlit = unlit.filter(|_| lighting != 0 && gbuffer != 0);
        Config {
            skip,
            unlit,
            lighting,
            gbuffer,
            gain: var("VICTORY_L7_UNLIT_GAIN").and_then(|v| v.parse().ok()).unwrap_or(1.0),
            albedo: var("VICTORY_L7_UNLIT_ALBEDO").and_then(|v| v.parse().ok()).unwrap_or(0),
            decode: match var("VICTORY_L7_UNLIT_DECODE").as_deref() {
                Some("linear") => Decode::Linear,
                Some("square") => Decode::Square,
                Some("gamma") => Decode::Gamma,
                _ => Decode::Auto,
            },
            far: match var("VICTORY_L7_UNLIT_FAR").as_deref() {
                Some("off" | "none") => None,
                Some(v) => Some(v.parse().unwrap_or(0.0)),
                None => Some(0.0),
            },
        }
    })
}

/// Logs the switches, once (at the first submission).
pub(crate) fn announce() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let r = roles();
        let c = config();
        let skipped: Vec<&str> = r
            .names
            .iter()
            .enumerate()
            .filter(|(i, _)| c.skip & 1 << i != 0)
            .map(|(_, n)| n.as_str())
            .collect();
        tracing::info!(
            roles = r.by_hash.len(),
            role_names = r.names.len(),
            skip = ?skipped,
            unlit = ?c.unlit,
            gain = c.gain,
            albedo = c.albedo,
            decode = ?c.decode,
            far = ?c.far,
            "lot 7 (unlit mode) switches: VICTORY_L7_ROLES, VICTORY_L7_SKIP, VICTORY_L7_UNLIT(_GAIN/_ALBEDO/_DECODE/_FAR)"
        );
    });
}

/// The role, among those `VICTORY_L7_SKIP` names, of a pipeline whose
/// shaders have the roles `mask`.
pub(crate) fn skipped_role(mask: u64) -> Option<&'static str> {
    let hit = mask & config().skip;
    (hit != 0).then(|| roles().names[hit.trailing_zeros() as usize].as_str())
}

/// What `VICTORY_L7_UNLIT` asks, if the lighting pipeline with roles `mask`
/// is one it acts on.
pub(crate) fn unlit_for(mask: u64) -> Option<Unlit> {
    let c = config();
    c.unlit.filter(|_| mask & c.lighting != 0)
}

/// Whether draws with roles `mask` fill the G-buffer the albedo is taken
/// from (with `VICTORY_L7_UNLIT` only).
pub(crate) fn fills_gbuffer(mask: u64) -> bool {
    let c = config();
    c.unlit.is_some() && mask & c.gbuffer != 0
}

/// A render target of a G-buffer draw: the title texture, the format of
/// its view, the view's mip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Slot {
    pub(crate) resource: usize,
    pub(crate) format: i32,
    pub(crate) mip: u32,
}

/// A set of G-buffer targets and their size (that of the first).
#[derive(Clone, Debug)]
struct GBuffer {
    slots: Vec<Slot>,
    width: u64,
    height: u32,
}

/// The G-buffers seen, the latest first (a few: the main view's, mirrors').
static GBUFFERS: Mutex<Vec<GBuffer>> = Mutex::new(Vec::new());

/// Notes the targets of a G-buffer draw.
pub(crate) fn note_gbuffer(slots: Vec<Slot>) {
    let Some(first) = slots.first() else { return };
    let Some(texture) = crate::resource::host_resource(first.resource as This) else {
        return;
    };
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let mut seen = GBUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(i) = seen.iter().position(|g| g.slots == slots) {
        let g = seen.remove(i);
        seen.insert(0, g);
        return;
    }
    let described: Vec<String> = slots
        .iter()
        .map(|s| {
            let size = crate::resource::host_resource(s.resource as This).map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{}", d.Width, d.Height)
            });
            format!("{:#x} {} f{} mip {}", s.resource, size.unwrap_or_default(), s.format, s.mip)
        })
        .collect();
    tracing::info!(targets = ?described, "lot 7: G-buffer targets seen (VICTORY_L7_UNLIT_ALBEDO picks one by its place here)");
    seen.insert(
        0,
        GBuffer {
            width: d.Width >> first.mip,
            height: d.Height >> first.mip,
            slots,
        },
    );
    seen.truncate(8);
}

/// Composites recorded and lighting draws replaced since the last [`log`].
static COMPOSITES: AtomicU64 = AtomicU64::new(0);
pub(crate) static REPLACED: AtomicU64 = AtomicU64::new(0);

/// The per-frame counts, then zeroed (with the frame-rate line).
pub(crate) fn log(frames: f64) {
    let c = config();
    if c.unlit.is_none() {
        return;
    }
    let composites = COMPOSITES.swap(0, Relaxed) as f64 / frames.max(1.0);
    let replaced = REPLACED.swap(0, Relaxed) as f64 / frames.max(1.0);
    tracing::info!(
        "lot 7 per frame: {composites:.1} albedo composites (VICTORY_L7_UNLIT={:?}), {replaced:.1} lighting draws replaced",
        c.unlit
    );
}

/// What the lighting draw is bound to: its render targets' view formats
/// (null views included, as the host list has them), its first target, and
/// its depth view's format and texture if it has one.
pub(crate) struct Bound {
    pub(crate) formats: Vec<DXGI_FORMAT>,
    pub(crate) first: Option<usize>,
    pub(crate) depth: Option<(DXGI_FORMAT, usize)>,
}

/// Our pass's objects: the heap of its albedo views (a ring), and its
/// pipelines by targets.
struct Pass {
    views: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    next: std::sync::atomic::AtomicU32,
    root: ID3D12RootSignature,
    made: Mutex<Vec<(PsoKey, Option<ID3D12PipelineState>)>>,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

#[derive(Clone, PartialEq, Eq, Debug)]
struct PsoKey {
    formats: Vec<i32>,
    depth: i32,
    tested: bool,
    add: bool,
    samples: u32,
}

const VIEWS: u32 = 256;

const SOURCE: &str = r#"
cbuffer L7 : register(b0) {
    float gain;
    uint decode;
    float depth;
    float sx;
    float sy;
    float pad0;
    float pad1;
    float pad2;
};
Texture2D<float4> albedo : register(t0);
float4 corner(uint i : SV_VertexID) : SV_Position {
    return float4(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0, depth, 1.0);
}
float4 unlit(float4 p : SV_Position) : SV_Target0 {
    float4 a = albedo.Load(int3(int2(p.xy * float2(sx, sy)), 0));
    float3 c = saturate(a.rgb);
    if (decode == 1) {
        c = c * c;
    } else if (decode == 2) {
        c = pow(c, 2.2);
    }
    return float4(c * gain, 1.0);
}
"#;

fn pass() -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = crate::host::host()?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        let range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        };
        let table = D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        };
        let made = (|| -> windows::core::Result<Pass> {
            let root = crate::host::root_signature_from_params(
                &host.device,
                &[crate::host::root_param_constants(0, 8), table],
            )?;
            // SAFETY: plain heap creation and queries.
            unsafe {
                let views: ID3D12DescriptorHeap = host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: VIEWS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
                Ok(Pass {
                    cpu: views.GetCPUDescriptorHandleForHeapStart().ptr,
                    gpu: views.GetGPUDescriptorHandleForHeapStart().ptr,
                    step: host.device.GetDescriptorHandleIncrementSize(kind),
                    views,
                    next: std::sync::atomic::AtomicU32::new(0),
                    root,
                    made: Mutex::default(),
                })
            }
        })();
        match made {
            Ok(pass) => Some(pass),
            Err(e) => {
                tracing::warn!(error = %e, "lot 7: no root signature or heap for the unlit pass: VICTORY_L7_UNLIT does nothing");
                None
            }
        }
    })
    .as_ref()
}

impl Pass {
    fn pipeline(&self, key: &PsoKey) -> Option<ID3D12PipelineState> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, p)) = made.iter().find(|(k, _)| k == key) {
            return p.clone();
        }
        let p = match self.make(key) {
            Ok(p) => {
                tracing::info!(key = ?key, "lot 7: unlit pass pipeline made");
                Some(p)
            }
            Err(e) => {
                tracing::warn!(error = %e, key = ?key, "lot 7: the host refused the unlit pass pipeline for these targets");
                None
            }
        };
        made.push((key.clone(), p.clone()));
        p
    }

    fn make(&self, key: &PsoKey) -> windows::core::Result<ID3D12PipelineState> {
        use windows::core::s;
        let device = &crate::host::host().ok_or_else(windows::core::Error::empty)?.device;
        let vs = crate::host::compile(SOURCE, s!("corner"), s!("vs_5_1"))?;
        let ps = crate::host::compile(SOURCE, s!("unlit"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        let mut blend = D3D12_BLEND_DESC {
            AlphaToCoverageEnable: false.into(),
            IndependentBlendEnable: true.into(),
            ..Default::default()
        };
        let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
        for (i, &f) in key.formats.iter().enumerate().take(8) {
            formats[i] = DXGI_FORMAT(f);
            blend.RenderTarget[i] = D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: (i == 0 && key.add).into(),
                LogicOpEnable: false.into(),
                SrcBlend: D3D12_BLEND_ONE,
                DestBlend: D3D12_BLEND_ONE,
                BlendOp: D3D12_BLEND_OP_ADD,
                SrcBlendAlpha: D3D12_BLEND_ZERO,
                DestBlendAlpha: D3D12_BLEND_ONE,
                BlendOpAlpha: D3D12_BLEND_OP_ADD,
                LogicOp: D3D12_LOGIC_OP_NOOP,
                // The colour of the first target only: its alpha and the
                // other targets keep what they hold.
                RenderTargetWriteMask: if i == 0 { 0x7 } else { 0 },
            };
        }
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: borrowed for the call; the signature outlives it.
            pRootSignature: unsafe { std::mem::transmute_copy(&self.root) },
            VS: code(&vs[..]),
            PS: code(&ps[..]),
            BlendState: blend,
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: false.into(),
                ..Default::default()
            },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: key.tested.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                DepthFunc: D3D12_COMPARISON_FUNC_NOT_EQUAL,
                StencilEnable: false.into(),
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: key.formats.len().min(8) as u32,
            RTVFormats: formats,
            DSVFormat: DXGI_FORMAT(key.depth),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: key.samples,
                Quality: 0,
            },
            ..Default::default()
        };
        // SAFETY: the description points at live data.
        unsafe { device.CreateGraphicsPipelineState(&desc) }
    }
}

/// 8- and 10-bit UNORM formats (not sRGB): what `auto` squares.
fn unorm8(format: i32) -> bool {
    // R8G8B8A8_UNORM, B8G8R8A8_UNORM, B8G8R8X8_UNORM, R10G10B10A2_UNORM
    matches!(format, 28 | 87 | 88 | 24)
}

/// Records our pass into `list` before a lighting draw bound to `bound`:
/// the albedo of the latest G-buffer of the first target's size, times the
/// gain, into the first target's colour (added with `add`). The pass sets
/// the list's heaps, root signature, pipeline and topology; the caller puts
/// the title's back. False if nothing was recorded (logged once per reason).
pub(crate) fn composite(list: &crate::host::HostList, bound: &Bound, add: bool) -> bool {
    let c = config();
    let (Some(host), Some(pass)) = (crate::host::host(), pass()) else {
        return false;
    };
    let Some(first) = bound.first else {
        l7_once!("lot 7: a lighting draw without a first render target: no unlit pass");
        return false;
    };
    let Some(target) = crate::resource::host_resource(first as This) else {
        return false;
    };
    // SAFETY: a live resource.
    let t = unsafe { target.GetDesc() };
    if t.SampleDesc.Count != 1 {
        l7_once!(samples = t.SampleDesc.Count, "lot 7: a multisampled lighting target: no unlit pass");
        return false;
    }
    let gbuffer = {
        let seen = GBUFFERS.lock().unwrap_or_else(|e| e.into_inner());
        // The fullest of that size (a decal draw binds fewer targets: the
        // places would shift), the latest among equals.
        seen.iter()
            .filter(|g| g.width == t.Width && g.height == t.Height)
            .fold(None::<&GBuffer>, |best, g| match best {
                Some(b) if b.slots.len() >= g.slots.len() => Some(b),
                _ => Some(g),
            })
            .cloned()
    };
    let Some(gbuffer) = gbuffer else {
        l7_once!(width = t.Width, height = t.Height, "lot 7: no G-buffer of the lighting target's size seen yet: no unlit pass (until one is)");
        return false;
    };
    let Some(slot) = gbuffer.slots.get(c.albedo).copied() else {
        l7_once!(albedo = c.albedo, targets = gbuffer.slots.len(), "lot 7: VICTORY_L7_UNLIT_ALBEDO is past the G-buffer's targets: no unlit pass");
        return false;
    };
    if slot.resource == first {
        l7_once!("lot 7: the albedo target is the lighting target: no unlit pass");
        return false;
    }
    let Some(albedo) = crate::resource::host_resource(slot.resource as This) else {
        return false;
    };
    // SAFETY: a live resource.
    let a = unsafe { albedo.GetDesc() };
    if a.SampleDesc.Count != 1 || a.DepthOrArraySize != 1 || a.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
        l7_once!(desc = ?a, "lot 7: the albedo target is not a plain 2D texture: no unlit pass");
        return false;
    }
    let tested = c.far.is_some() && bound.depth.is_some();
    let key = PsoKey {
        formats: bound.formats.iter().map(|f| f.0).collect(),
        depth: bound.depth.map_or(0, |d| d.0.0),
        tested,
        add,
        samples: 1,
    };
    let Some(pipeline) = pass.pipeline(&key) else {
        return false;
    };
    let decode = match c.decode {
        Decode::Linear => 0u32,
        Decode::Square => 1,
        Decode::Gamma => 2,
        Decode::Auto => u32::from(unorm8(slot.format)),
    };
    // States: the albedo readable, the targets writable, the depth testable.
    const RENDER_TARGET: u32 = 0x4;
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const DEPTH_READ: u32 = 0x20;
    crate::list::ensure_state(list, slot.resource as This, SHADER_RESOURCE);
    crate::list::ensure_state(list, first as This, RENDER_TARGET);
    if let (true, Some((_, depth))) = (tested, bound.depth) {
        list.twin_use(depth as This, false, false);
        crate::list::ensure_state(list, depth as This, DEPTH_READ);
    }
    let (width, height) = ((a.Width >> slot.mip).max(1) as f32, (a.Height >> slot.mip).max(1) as f32);
    let consts: [u32; 8] = [
        c.gain.to_bits(),
        decode,
        c.far.unwrap_or(0.0).to_bits(),
        (width / t.Width as f32).to_bits(),
        (height / t.Height as f32).to_bits(),
        0,
        0,
        0,
    ];
    let index = pass.next.fetch_add(1, Relaxed) % VIEWS;
    let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr: pass.cpu + (index * pass.step) as usize,
    };
    let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
        ptr: pass.gpu + u64::from(index * pass.step),
    };
    let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT(slot.format),
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: slot.mip,
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    let l = &list.list;
    // SAFETY: an open list; a slot of our heap; live resources; our objects
    // live as long as the process.
    unsafe {
        host.device.CreateShaderResourceView(albedo, Some(&view), cpu);
        l.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        l.SetGraphicsRootSignature(&pass.root);
        l.SetPipelineState(&pipeline);
        l.SetGraphicsRoot32BitConstants(0, 8, consts.as_ptr().cast(), 0);
        l.SetGraphicsRootDescriptorTable(1, gpu);
        l.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(4));
        l.DrawInstanced(3, 1, 0, 0);
    }
    #[cfg(feature = "diagnostics")]
    list.note_op(|| {
        format!(
            "lot 7 unlit pass ({}) albedo {:#x} f{} into {first:#x} {}x{} formats {:?} depth {:?}",
            if add { "add" } else { "replace" },
            slot.resource,
            slot.format,
            t.Width,
            t.Height,
            key.formats,
            bound.depth.map(|d| d.0.0)
        )
    });
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        tracing::info!(
            albedo = format_args!("{:#x} f{} {}x{}", slot.resource, slot.format, a.Width, a.Height),
            target = format_args!("{first:#x} {}x{} formats {:?}", t.Width, t.Height, key.formats),
            depth = ?bound.depth.map(|d| d.0.0),
            tested,
            decode,
            gain = c.gain,
            add,
            "lot 7: first unlit pass recorded"
        );
    });
    COMPOSITES.fetch_add(1, Relaxed);
    true
}

#[cfg(test)]
mod tests {
    /// The pass's shaders compile with the system's compiler (Wine's under
    /// Proton) and the host makes its pipeline for targets like the tiled
    /// lighting's: a half-float colour pair over a depth-stencil target,
    /// tested or not, added or not.
    #[test]
    fn the_unlit_pass_pipeline_is_made_for_lighting_like_targets() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let pass = super::pass().expect("the pass's root signature and heap");
        // R16G16B16A16_FLOAT + R16G16_FLOAT over D32_FLOAT_S8X24_UINT; R16G16B16A16_FLOAT alone;
        // R11G11B10_FLOAT + R16G16_FLOAT over D24_UNORM_S8_UINT.
        for (formats, depth, tested, add) in [
            (vec![10, 34], 20, true, false),
            (vec![10], 0, false, true),
            (vec![26, 34], 45, true, true),
        ] {
            let key = super::PsoKey {
                formats,
                depth,
                tested,
                add,
                samples: 1,
            };
            assert!(pass.pipeline(&key).is_some(), "{key:?}");
        }
    }
}
