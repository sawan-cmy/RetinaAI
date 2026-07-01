"use client"

import { useEffect, useRef } from "react"
import * as THREE from "three"

export function MedicalModelCanvas() {
  const mountRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(55, mount.clientWidth / mount.clientHeight, 0.1, 100)
    camera.position.z = 4.4

    // ponytail: Preserve the buffer for browser pixel QA; drop it if this hero becomes a heavy scene.
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(mount.clientWidth, mount.clientHeight)
    mount.appendChild(renderer.domElement)

    const group = new THREE.Group()
    scene.add(group)

    const retina = new THREE.Mesh(
      new THREE.TorusGeometry(1.35, 0.08, 20, 140),
      new THREE.MeshBasicMaterial({ color: 0x14b8a6, transparent: true, opacity: 0.9 }),
    )
    group.add(retina)

    const disc = new THREE.Mesh(
      new THREE.CircleGeometry(1.18, 96),
      new THREE.MeshBasicMaterial({ color: 0x2563eb, transparent: true, opacity: 0.11 }),
    )
    group.add(disc)

    const vesselMaterial = new THREE.LineBasicMaterial({ color: 0xf8fafc, transparent: true, opacity: 0.46 })
    for (let i = 0; i < 18; i += 1) {
      const angle = (i / 18) * Math.PI * 2
      const points = [
        new THREE.Vector3(0, 0, 0.02),
        new THREE.Vector3(Math.cos(angle) * 0.45, Math.sin(angle) * 0.45, 0.02),
        new THREE.Vector3(Math.cos(angle + 0.16) * 1.05, Math.sin(angle + 0.16) * 1.05, 0.02),
      ]
      group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), vesselMaterial))
    }

    const scan = new THREE.Mesh(
      new THREE.PlaneGeometry(2.7, 0.025),
      new THREE.MeshBasicMaterial({ color: 0x14b8a6, transparent: true, opacity: 0.8 }),
    )
    group.add(scan)

    const onResize = () => {
      if (!mount) return
      camera.aspect = mount.clientWidth / mount.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(mount.clientWidth, mount.clientHeight)
    }
    window.addEventListener("resize", onResize)

    let frame = 0
    let animationId = 0
    const animate = () => {
      frame += 0.01
      group.rotation.z = Math.sin(frame) * 0.08
      group.rotation.y = Math.sin(frame * 0.7) * 0.12
      scan.position.y = Math.sin(frame * 2) * 1.05
      renderer.render(scene, camera)
      animationId = window.requestAnimationFrame(animate)
    }
    animate()

    return () => {
      window.cancelAnimationFrame(animationId)
      window.removeEventListener("resize", onResize)
      renderer.dispose()
      mount.removeChild(renderer.domElement)
    }
  }, [])

  return <div id="medical-model-canvas" ref={mountRef} className="h-full min-h-[320px] w-full" aria-label="Animated retinal AI model" />
}

